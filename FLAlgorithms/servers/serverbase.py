import functools
from collections import defaultdict
import heapq
import math
import hdbscan
import torch
import os
import numpy as np
import h5py
import copy
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

from FLAlgorithms.functions.miloss import Mine
import random
import torch.nn.functional as F
import core
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split


class Server:
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda,
                 num_glob_iters, local_epochs, optimizer, num_users, times, fo, current_time, malnum, malclient,
                 poisonratio, poison_label, attack_method, per_epoch, defense):

        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.defense = defense
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)  # 复制模型
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_global_train_acc, self.rs_global_train_loss, self.rs_global_test_acc, self.rs_local_train_acc_per, self.rs_local_train_loss_per, self.rs_local_test_acc_per = [], [], [], [], [], []
        self.rs_global_train_asr, self.rs_global_train_asr_loss, self.rs_global_test_asr, self.rs_local_train_asr_per, self.rs_local_train_asr_loss_per, self.rs_local_test_asr_per = [], [], [], [], [], []
        self.times = times
        self.fo = fo
        self.current_time = current_time
        self.malnum = malnum
        self.malclient = malclient
        self.poisonratio = poisonratio
        self.poisonlabel = poison_label
        self.attack_method = attack_method
        self.folder_path = f'results/{self.dataset}_{current_time}_{algorithm}_{attack_method}_{defense}_{poisonratio}_{per_epoch}'
        self.mi_path = f'results/{self.dataset}_{current_time}_{algorithm}_{attack_method}_{poisonratio}_{per_epoch}/'
        self.savedmodelpath = f'saved_model'

        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            print('Folder already exists')

    def aggregate_grads(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"
        for user in self.users:
            user.set_parameters(self.model)

    def send_pmodel_parma(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"
        for user in self.users:
            user.set_parameters(user.pmodel)

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            newvalue = server_param.data + user_param.data.clone() * ratio
            server_param.data.copy_(newvalue)

    def model_dist_norm(self, user):
        squared_sum = 0
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            squared_sum += torch.sum(torch.pow(server_param.data.clone() - user_param.data.clone(), 2))

        return math.sqrt(squared_sum)

    def aggregate_parameters(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"

        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0

        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            # self.add_parameters(user, 1/ len(self.selected_users))

    def Trimmed_Mean(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"

        clients_params = []
        for user in self.selected_users:
            clients_params.append(
                np.concatenate([param.data[:].cpu().numpy().flatten() for param in user.get_parameters()]))
        clients_params = torch.tensor(np.array(clients_params))

        m = 0
        for user in self.selected_users:
            if user.id in self.malclient:
                m += 1

        a = clients_params.sort(dim=0)[0][m:len(self.selected_users) - m]
        b = torch.mean(a, dim=0)

        # 全局模型归零
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)

        # 模型参数更新
        offset = 0
        for server_param in self.model.parameters():
            with torch.no_grad():
                new_size = functools.reduce(lambda x, y: x * y, server_param.shape)
                new_value = b[offset:offset + new_size]
                server_param.data[:] = new_value.clone().detach().reshape(server_param.shape)
                offset += new_size

    def Multi_Krum(self):
        def krum_create_distances(clients_params):
            distances = defaultdict(dict)
            for i in range(len(clients_params)):
                for j in range(i):
                    distances[i][j] = distances[j][i] = np.linalg.norm(clients_params[i] - clients_params[j])
            return distances

        assert self.users is not None and len(self.users) > 0, "用户列表为空"
        clients_params = []
        for user in self.selected_users:
            clients_params.append(
                np.concatenate([param.data[:].cpu().numpy().flatten() for param in user.get_parameters()]))
        clients_params = np.array(clients_params)

        m = 0
        for user in self.selected_users:
            if user.id in self.malclient:
                m += 1
        non_malicious_count = len(self.selected_users) - m

        distances = krum_create_distances(clients_params)

        selection_set = []
        krumscore = []

        for user in distances.keys():
            errors = sorted(distances[user].values())
            current_error = sum(errors[:non_malicious_count])
            krumscore.append(current_error)
        tmp = krumscore[0]
        krumscore[0] = krumscore[1]
        krumscore[1] = tmp
        result = map(krumscore.index, heapq.nsmallest(non_malicious_count, krumscore))
        for i in result:
            selection_set.append(clients_params[i])

        result_params = torch.tensor(selection_set).mean(dim=0)

        # 全局模型归零
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)

        # 模型参数更新
        offset = 0
        for server_param in self.model.parameters():
            with torch.no_grad():
                new_size = functools.reduce(lambda x, y: x * y, server_param.shape)
                new_value = result_params[offset:offset + new_size]
                server_param.data[:] = new_value.clone().detach().reshape(server_param.shape)
                offset += new_size

    def save_model(self):
        model_path = os.path.join(f'{self.folder_path}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def save_model_epoch(self, global_iteration):
        model_path = os.path.join(f'{self.folder_path}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), os.path.join(model_path, str(global_iteration) + "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join(f'{self.savedmodelpath}', "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def load_model_pretrain(self, method):
        model_path = os.path.join(f'{self.savedmodelpath}', str(method) + "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join(f'{self.folder_path}', self.dataset, "server" + ".pt"))

    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if (num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        return np.random.choice(self.users, num_users, replace=False)

    # define function for persionalized agegatation.
    def persionalized_update_parameters(self, user, ratio):
        # only argegate the local_weight_update
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def persionalized_aggregate_parameters(self):
        assert self.users is not None and len(self.users) > 0, "用户列表为空"

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        # if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            # self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta) * pre_param.data + self.beta * param.data

    def persionalized_Multi_Krum(self):
        previous_param = copy.deepcopy(list(self.model.parameters()))

        self.Multi_Krum()

        # aggregate avergage model with previous model using parameter beta
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta) * pre_param.data + self.beta * param.data

    def persionalized_Trimmed_Mean(self):
        previous_param = copy.deepcopy(list(self.model.parameters()))

        self.Trimmed_Mean()

        # aggregate avergage model with previous model using parameter beta
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta) * pre_param.data + self.beta * param.data

    def save_results(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(
            self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
        if (self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_global_test_acc) != 0 & len(self.rs_global_train_acc) & len(self.rs_global_train_loss)):
            with h5py.File('{}'.format(self.folder_path) + '/' + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_global_test_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_global_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_global_train_loss)
                hf.close()

        # store persionalized value
        alg = self.dataset + "_" + self.algorithm + "_p"
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(
            self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
        if (self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_local_test_acc_per) != 0 & len(self.rs_local_train_acc_per) & len(
                self.rs_local_train_loss_per)):
            with h5py.File('{}'.format(self.folder_path) + '/' + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_local_test_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_local_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_local_train_loss_per)
                hf.close()

    def save_poison_results(self):
        alg = self.dataset + "_" + self.algorithm + "_poison"
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(
            self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)

        if (self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)

        alg = alg + "_" + str(self.times)

        if (len(self.rs_global_test_asr) != 0 & len(self.rs_global_train_asr) & len(self.rs_global_train_asr_loss)):
            with h5py.File('{}'.format(self.folder_path) + '/' + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_asr', data=self.rs_global_test_asr)
                hf.create_dataset('rs_train_asr', data=self.rs_global_train_asr)
                hf.create_dataset('rs_train_asr_loss', data=self.rs_global_train_asr_loss)
                hf.close()

        # store persionalized value
        alg = self.dataset + "_" + self.algorithm + "_p" + "_poison"
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(
            self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)

        if (self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)

        alg = alg + "_" + str(self.times)

        if (len(self.rs_local_test_asr_per) != 0 & len(self.rs_local_train_asr_per) & len(
                self.rs_local_train_asr_loss_per)):
            with h5py.File('{}'.format(self.folder_path) + '/' + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_asr', data=self.rs_local_test_asr_per)
                hf.create_dataset('rs_train_asr', data=self.rs_local_train_asr_per)
                hf.create_dataset('rs_train_asr_loss', data=self.rs_local_train_asr_loss_per)

    def test(self):
        '''tests self.latest_model on given clients
        在所有用户上用自己的数据基于自己的本地模型测试
        '''
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def poison_test(self, poiosnlabel, trigger, pattern):
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.poisontest(poiosnlabel, trigger, pattern)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss(self):
        # 所有用户在训练数据集上的acc、loss、id、样本数，基于自己模型
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    def poison_train_error_and_loss(self, poiosnlabel, trigger, pattern):
        # 所有用户在训练数据集上的acc、loss、id、样本数，基于自己模型
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.poison_train_error_and_loss(poiosnlabel=poiosnlabel, trigger=trigger, pattern=pattern)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def test_persionalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.test_persionalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss_persionalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate(self):
        stats = self.test()  # 每个用户的基于全局模型、测试数据的准确率，样本数、id
        stats_train = self.train_error_and_loss()  # 每个用户的基于全局模型、训练数据的准确率，样本数、id、loss
        global_test_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        global_train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        global_train_loss = 0

        print("Average Global Accurancy: ", global_test_acc)
        print("Average Global Trainning Accurancy: ", global_train_acc)
        print()

        mal_person_acc = []
        ben_person_acc = []
        for i in range(len(stats[0])):
            if stats[0][i] in self.malclient:
                mal_person_acc.append(stats[2][i] * 1.0 / stats[1][i])
            else:
                ben_person_acc.append(stats[2][i] * 1.0 / stats[1][i])

        print("global benign client acc list:{}; mean acc:{}".format(ben_person_acc,
                                                                     sum(ben_person_acc) / len(ben_person_acc)))
        print("global malicious client acc list:{}; mean acc:{}".format(mal_person_acc,
                                                                        sum(mal_person_acc) / len(mal_person_acc)))

        return global_test_acc, global_train_acc, global_train_loss, sum(ben_person_acc) / len(ben_person_acc), sum(
            mal_person_acc) / len(mal_person_acc)

    def poison_evaluate(self, trigger, pattern):
        # 每个用户的基于全局模型、测试数据的准确率，样本数、id
        stats = self.poison_test(trigger=trigger, poiosnlabel=self.poisonlabel, pattern=pattern)
        # 每个用户的基于全局模型、训练数据的准确率，样本数、id、loss
        stats_train = self.poison_train_error_and_loss(trigger=trigger, poiosnlabel=self.poisonlabel, pattern=pattern)

        global_test_asr = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        global_train_asr = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        global_train_asr_loss = 0

        print("Average Global ATTACK ALL ASR: ", global_test_asr)
        print("Average Global ATTACK ALL train ASR: ", global_train_asr)

        mal_person_asr = []
        ben_person_asr = []
        for i in range(len(stats[0])):
            if stats[0][i] in self.malclient:
                mal_person_asr.append(stats[2][i] * 1.0 / stats[1][i])
            else:
                if stats[1][i] != 0:
                    ben_person_asr.append(stats[2][i] * 1.0 / stats[1][i])

        print("global benign client asr list:{}; mean asr:{}".format(ben_person_asr,
                                                                     sum(ben_person_asr) / len(ben_person_asr)))
        print("global malicious client asr list:{}; mean asr:{}".format(mal_person_asr,
                                                                        sum(mal_person_asr) / len(mal_person_asr)))

        return global_test_asr, global_train_asr, global_train_asr_loss, sum(ben_person_asr) / len(ben_person_asr), \
               sum(mal_person_asr) / len(mal_person_asr)

    def evaluate_personalized_model(self):
        stats = self.test_persionalized_model()
        stats_train = self.train_error_and_loss_persionalized_model()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        # print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        # print("Average Personal Trainning Loss: ", train_loss)

    def evaluate_one_step(self, per_epoch, trigger, pattern):
        for c in self.users:
            c.train_one_step(per_epoch)  # 每个用户训练一次，表示本地 fine-tune

        stats = self.test()
        stats_train = self.train_error_and_loss()

        # trigger is a list
        poison_stats = self.poison_test(trigger=trigger, poiosnlabel=self.poisonlabel, pattern=pattern)
        poison_stats_train = self.poison_train_error_and_loss(trigger=trigger, poiosnlabel=self.poisonlabel,
                                                              pattern=pattern)

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)  # 本地 fine-tune 结束后，恢复之前的模型，重新训练

        per_local_test_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        per_localtrain_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        per_localtrain_loss = 0
        print("Average Personal Accurancy (k local SGD): ", per_local_test_acc)
        print("Average Personal Trainning Accurancy (k local SGD): ", per_localtrain_acc)

        mal_person_acc = []
        ben_person_acc = []
        for i in range(len(stats[0])):
            if stats[0][i] in self.malclient:
                mal_person_acc.append(stats[2][i] * 1.0 / stats[1][i])
            else:
                ben_person_acc.append(stats[2][i] * 1.0 / stats[1][i])

        print("global benign client acc list:{}; mean acc:{}".format(ben_person_acc,
                                                                     sum(ben_person_acc) / len(ben_person_acc)))
        print("global malicious client acc list:{}; mean acc:{}".format(mal_person_acc,
                                                                        sum(mal_person_acc) / len(mal_person_acc)))

        per_local_test_asr = np.sum(poison_stats[2]) * 1.0 / np.sum(poison_stats[1])
        per_localtrain_asr = np.sum(poison_stats_train[2]) * 1.0 / np.sum(poison_stats_train[1])
        per_localtrain_losssr = 0
        print("Average Personal ATTACK ALL ASR (k local SGD): ", per_local_test_asr)
        print("Average Personal ATTACK ALL Trainning ASR (k local SGD): ", per_localtrain_asr)

        mal_person_asr = []
        ben_person_asr = []
        for i in range(len(poison_stats[0])):
            if poison_stats[0][i] in self.malclient:
                mal_person_asr.append(poison_stats[2][i] * 1.0 / poison_stats[1][i])
            else:
                if poison_stats[1][i] != 0:
                    ben_person_asr.append(poison_stats[2][i] * 1.0 / poison_stats[1][i])

        print("person benign client asr list:{}; mean asr:{}".format(ben_person_asr,
                                                                     sum(ben_person_asr) / len(ben_person_asr)))
        print("person malicious client asr list:{}; mean asr:{}".format(mal_person_asr,
                                                                        sum(mal_person_asr) / len(mal_person_asr)))

        return per_local_test_acc, per_localtrain_acc, per_localtrain_loss, per_local_test_asr, per_localtrain_asr, per_localtrain_losssr, \
               sum(ben_person_asr) / len(ben_person_asr), sum(mal_person_asr) / len(mal_person_asr), sum(
            ben_person_acc) / len(ben_person_acc), sum(mal_person_acc) / len(mal_person_acc)

    def evaluate_one_step_poison(self, per_epoch, trigger, pattern):
        for c in self.users:
            if c.id in self.malclient:
                c.train_one_step_poison(per_epoch, trigger=trigger, pattern=pattern, poison_label=self.poisonlabel,
                                        poison_ratio=self.poisonratio)
            else:
                c.train_one_step(per_epoch)  # 每个用户训练一次，表示本地 fine-tune

        stats = self.test()
        stats_train = self.train_error_and_loss()

        # trigger is a list
        poison_stats = self.poison_test(trigger=trigger, poiosnlabel=self.poisonlabel, pattern=pattern)
        poison_stats_train = self.poison_train_error_and_loss(trigger=trigger, poiosnlabel=self.poisonlabel,
                                                              pattern=pattern)

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)  # 本地 fine-tune 结束后，恢复之前的模型，重新训练

        per_local_test_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        per_localtrain_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        per_localtrain_loss = 0
        print("Average Personal Accurancy (k local SGD): ", per_local_test_acc)
        print("Average Personal Trainning Accurancy (k local SGD): ", per_localtrain_acc)
        mal_person_acc = []
        ben_person_acc = []
        for i in range(len(stats[0])):
            if stats[0][i] in self.malclient:
                mal_person_acc.append(stats[2][i] * 1.0 / stats[1][i])
            else:
                ben_person_acc.append(stats[2][i] * 1.0 / stats[1][i])

        print("global benign client acc list:{}; mean acc:{}".format(ben_person_acc,
                                                                     sum(ben_person_acc) / len(ben_person_acc)))
        print("global malicious client acc list:{}; mean acc:{}".format(mal_person_acc,
                                                                        sum(mal_person_acc) / len(mal_person_acc)))

        per_local_test_asr = np.sum(poison_stats[2]) * 1.0 / np.sum(poison_stats[1])
        per_localtrain_asr = np.sum(poison_stats_train[2]) * 1.0 / np.sum(poison_stats_train[1])
        per_localtrain_losssr = 0
        print("Average Personal ATTACK ALL ASR (k local SGD): ", per_local_test_asr)
        print("Average Personal ATTACK ALL Trainning ASR (k local SGD): ", per_localtrain_asr)

        mal_person_asr = []
        ben_person_asr = []
        for i in range(len(poison_stats[0])):
            if poison_stats[0][i] in self.malclient:
                mal_person_asr.append(poison_stats[2][i] * 1.0 / poison_stats[1][i])
            else:
                if poison_stats[1][i] != 0:
                    ben_person_asr.append(poison_stats[2][i] * 1.0 / poison_stats[1][i])

        print("person benign client asr list:{}; mean asr:{}".format(ben_person_asr,
                                                                     sum(ben_person_asr) / len(ben_person_asr)))
        print("person malicious client asr list:{}; mean asr:{}".format(mal_person_asr,
                                                                        sum(mal_person_asr) / len(mal_person_asr)))

        return per_local_test_acc, per_localtrain_acc, per_localtrain_loss, per_local_test_asr, per_localtrain_asr, per_localtrain_losssr, \
               sum(ben_person_asr) / len(ben_person_asr), sum(mal_person_asr) / len(mal_person_asr), sum(
            ben_person_acc) / len(ben_person_acc), sum(
            mal_person_acc) / len(mal_person_acc)

    def trigger_evasion_mnist(self, model, trigger, glob_iter, attackstart):
        models = copy.deepcopy(model)
        models.eval()
        init = False
        pre_trigger = torch.tensor(trigger[0]).cuda()
        new_trigger_list = []

        dataset = []

        for labelindex in range(10):
            count = 1
            for user in self.users:
                if user.id in self.malclient:
                    for X, Y in user.trainloaderfull:
                        for i in range(len(X)):
                            if Y[i] == labelindex and count < 100:
                                dataset.append((X[i], Y[i]))
                                count += 1
                            if count >= 100:
                                break
                        if count >= 100:
                            break
                if count >= 100:
                    break

        dataloaders = DataLoader(dataset, batch_size=64, shuffle=True)

        iter_dataloader = iter(dataloaders)

        def get_batch():
            try:  # Samples a new batch for persionalizing
                (X, y) = next(iter_dataloader)
            except StopIteration:
                iter_trainloader = iter(dataloaders)
                (X, y) = next(iter_trainloader)
            return (X.to(self.device), y.to(self.device))

        for e in range(1, 51, 1):  # learning loss 51,  17
            corrects = 0
            datas, labels = get_batch()  # 取出一个batch数据
            x = Variable(datas)
            y = Variable(labels)
            y_target = torch.LongTensor(y.size()).fill_(1)
            y_target = Variable(y_target, requires_grad=False).to(self.device)
            if not init:
                noise = copy.deepcopy(pre_trigger)
                noise = Variable(noise, requires_grad=True).to(self.device)
                init = True

            #image对应位置先置0，再加 noise
            for index in range(0, len(x)):
                for i in range(10, 20, 1):
                    for j in range(10, 20, 1):
                        x[index][0][i][j] = 0

            output = model((x + noise).float())
            classloss = nn.functional.cross_entropy(output, y_target)

            loss = classloss
            model.zero_grad()
            if noise.grad:
                noise.grad.fill_(0)
            loss.backward(retain_graph=True)

            noise = noise - noise.grad * 0.1

            for i in range(28):
                for j in range(28):
                    if i in range(10, 20, 1) and j in range(10, 20, 1):
                        continue
                    else:
                        noise[0][i][j] = 0

            noise = torch.clamp(noise, -1, 1)
            noise = Variable(noise, requires_grad=True).to(self.device)
            pred = output.data.max(1)[1]
            correct = torch.eq(pred, y_target).float().mean().item()
            corrects += pred.eq(y_target.data.view_as(pred)).cpu().sum().item()

            print('batchid:{},correct:{},noise:{}'.format(e, correct * 100, noise.data.norm()))

        for i in range(10):
            new_trigger_list.append(copy.deepcopy(noise))

        return new_trigger_list

    def trigger_all_mnist_l2(self, trigger, pattern, attackstart, intinal_trigger, glob_iter):
        models = copy.deepcopy(self.model)
        models.eval()
        net_parameters = list(models.parameters())
        noise = copy.deepcopy(trigger[0]).to(self.device)
        noise = Variable(noise, requires_grad=True)
        new_trigger_list = []

        dataset = []
        for labelindex in range(10):
            count = 1
            for user in self.users:
                if user.id in self.malclient:
                    for X, Y in user.trainloaderfull:
                        for i in range(len(X)):
                            if Y[i] == labelindex and count < 100:
                                dataset.append((X[i], Y[i]))
                                count += 1
                            if count >= 100:
                                break
                        if count >= 100:
                            break
                if count >= 100:
                    break

        dataloaders = DataLoader(dataset, batch_size=64, shuffle=True)

        noisetemp = torch.zeros((1, 28, 28)).float().to(self.device)

        for i in range(0, len(pattern)):
            pos = pattern[i]
            noisetemp[0][pos[0]][pos[1]] = 1

        round = 30 + 1
        if self.algorithm == "PerAvg-HF":
            round = 10 + 1

        for e in range(1, round, 1):
            total_loss = 0
            for batch_id, (datas, labels) in enumerate(dataloaders):
                x = Variable(datas.to(self.device))
                y = Variable(labels.to(self.device))

                y_target = torch.LongTensor(y.size()).fill_(int(self.poisonlabel))
                y_target = Variable(y_target.to(self.device), requires_grad=False)

                for param in list(models.parameters()):
                    param.requires_grad = True

                output_nor = models((x).float())
                loss_nor = nn.functional.cross_entropy(output_nor, y)
                grad_nor = torch.autograd.grad(loss_nor, net_parameters)
                grad_nor = list((_.detach().clone() for _ in grad_nor))

                #image 对应的pattern位置先置 0， 后加 noise
                patterntensor = torch.ones((1, 28, 28)).float().to(self.device)
                for i in range(0, len(pattern)):
                    pos = pattern[i]
                    patterntensor[0][pos[0]][pos[1]] = 0

                patterntensor = patterntensor.unsqueeze(0)
                patterntensor = patterntensor.repeat(len(datas), 1, 1, 1)

                x = x * patterntensor

                output_mal = models((x + noise).float())
                loss_mal = nn.functional.cross_entropy(output_mal, y_target)
                grad_mal = torch.autograd.grad(loss_mal, net_parameters, create_graph=True)

                #gradient matching
                loss = self.match_l2_loss(grad_mal, grad_nor)
                total_loss += loss.item()

                models.zero_grad()
                if noise.grad:
                    noise.grad.fill_(0)

                loss.backward(retain_graph=True)
                noise = noise - noise.grad * 0.1
                noise = noise * noisetemp
                noise = torch.clamp(noise, -1, 1)

                noise = Variable(noise.data, requires_grad=True)

            print('l2 loss:{}'.format(total_loss))

        for i in range(10):
            new_trigger_list.append(copy.deepcopy(noise))

        return new_trigger_list


    def proj_lp(self, v, xi, p):
        # Project on the lp ball centered at 0 and of radius xi
        # SUPPORTS only p = 2 and p = Inf for now
        if p == 2:
            v = v * min(1, xi / torch.norm(v))
            # v = v / np.linalg.norm(v.flatten(1)) * xi
        elif p == np.inf:
            v = np.sign(v) * np.minimum(abs(v), xi)
        else:
            raise ValueError('Values of p different from 2 and Inf are currently not supported...')
        return v

    def match_loss(self, grad_mal, grad_nor):
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(grad_nor)):
            gw_real_vec.append(grad_nor[ig].reshape((-1)))
            gw_syn_vec.append(grad_mal[ig].reshape((-1)))

        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

        return dis

    def match_l2_loss(self, grad_mal, grad_nor):
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(grad_nor)):
            gw_real_vec.append(grad_nor[ig].reshape((-1)))
            gw_syn_vec.append(grad_mal[ig].reshape((-1)))

        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)

        dis = torch.sqrt(torch.sum((gw_syn_vec - gw_real_vec) ** 2))

        return dis

    def mi_loss(self, grad_mal, grad_nor, mi_model):
        mi_model.eval()
        gw_nor_vec = []
        gw_mal_vec = []
        for ig in range(len(grad_mal)):
            gw_nor_vec.append(grad_mal[ig].reshape((-1)))
            gw_mal_vec.append(grad_nor[ig].reshape((-1)))

        gw_nor_vec = torch.cat(gw_nor_vec, dim=0)
        gw_mal_vec = torch.cat(gw_mal_vec, dim=0)
        mi_lb, t, et = self.mutual_information(gw_nor_vec, gw_mal_vec, mi_model)

        return mi_lb

    def learn_mine(self, x, y, mine_net, mine_net_optim, ma_et, ma_rate=0.0001):
        # 梯度进行处理
        gw_nor_vec = []
        gw_mal_vec = []
        for ig in range(len(x)):
            gw_nor_vec.append(x[ig].reshape((-1)))
            gw_mal_vec.append(y[ig].reshape((-1)))

        gw_nor_vec = torch.cat(gw_nor_vec, dim=0)

        gw_mal_vec = torch.cat(gw_mal_vec, dim=0)

        mi_lb, t, et = self.mutual_information(gw_nor_vec, gw_mal_vec, mine_net)
        ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)

        # unbiasing use moving average
        # loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
        # use biased estimator
        loss = - mi_lb

        mine_net_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(mine_net.parameters(), max_norm=20, norm_type=2)
        mine_net_optim.step()
        return mi_lb, ma_et

    def mutual_information(self, joint, marginal, mine_net):
        t = mine_net(joint)
        et = torch.exp(mine_net(marginal))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et))
        return mi_lb, t, et

    def train_mine_estimator(self, grad_nor, grad_mal, mine_net, mine_net_optim, batch_size=100, iter_num=int(5e+3),
                             log_freq=int(1e+3)):
        # data is grad_nor and grad_mal
        result = list()
        ma_et = 1.
        for i in range(iter_num):
            mi_lb, ma_et = self.learn_mine(grad_nor, grad_mal, mine_net, mine_net_optim, ma_et)
            result.append(mi_lb.detach().cpu().numpy())
            if (i + 1) % (log_freq) == 0:
                print(result[-1])
        return result

    def compute_grad_mask(self, models, ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model = copy.deepcopy(models)
        model.train()
        model.zero_grad()

        dataset = []
        for user in self.users:
            if user.id in self.malclient:
                for X, Y in user.trainloaderfull:
                    for i in range(len(X)):
                        dataset.append((X[i], Y[i]))
        dataloaders = DataLoader(dataset, self.batch_size)

        for batch_id, (datas, labels) in enumerate(dataloaders):
            input = datas.to(self.device)
            label = labels.to(self.device)
            output = model(input)
            loss = nn.functional.cross_entropy(output, label)
            loss.backward(retain_graph=True)

        mask_grad_list = []
        grad_list = []
        grad_abs_sum_list = []
        k_layer = 0
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                grad_list.append(parms.grad.abs().view(-1))
                grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())
                k_layer += 1
        grad_list = torch.cat(grad_list).cuda()
        _, indices = torch.topk(-1 * grad_list, int(len(grad_list) * ratio))  # 保留
        mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
        mask_flat_all_layer[indices] = 1.0
        count = 0
        percentage_mask_list = []
        k_layer = 0
        grad_abs_percentage_list = []
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                gradients_length = len(parms.grad.abs().view(-1))
                mask_flat = mask_flat_all_layer[count:count + gradients_length].cuda()
                mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
                count += gradients_length
                percentage_mask1 = mask_flat.sum().item() / float(gradients_length) * 100.0
                percentage_mask_list.append(percentage_mask1)
                grad_abs_percentage_list.append(grad_abs_sum_list[k_layer] / np.sum(grad_abs_sum_list))
                k_layer += 1

        model.zero_grad()

        return mask_grad_list

    def apply_grad_mask(self, model, mask_grad_list):
        mask_grad_list_copy = iter(mask_grad_list)
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                parms.grad = parms.grad * next(mask_grad_list_copy)
