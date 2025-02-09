import copy
import torch
import utils.csv_record as csv_record
from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np


# Implementation for FedAvg Server

class FedAvg(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times, fo, current_time, malnum, malclient, poisonratio,
                 poison_label, attack_method, per_epoch, defense):
        super().__init__(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times, fo, current_time, malnum, malclient, poisonratio,
                         poison_label, attack_method, per_epoch, defense)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])  # 总用户
        for i in range(total_users):
            id, train, test = read_user_data(i, data, dataset)  # 取出用户i的数据集
            user = UserAVG(device, id, train, test, model, dataset, batch_size, learning_rate, beta, lamda,
                           local_epochs)  # 生成user对象
            self.users.append(user)  # 将user对象放到server的user列表里
            self.total_train_samples += user.train_samples

        print("Number of users / total users:", num_users, " / ", total_users)
        print("Finished creating FedAvg server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self, trigger, per_epoch, attack_start, pattern, oneshot, clip_rate, defense):
        original_trigger_list = trigger
        optimized_trigger_list = copy.deepcopy(original_trigger_list)
        First_Attack = False

        for glob_iter in range(0, self.num_glob_iters + 1):
            print("-------------Round number: ", glob_iter, " -------------")

            self.send_parameters()  # 将当前的全局模型分给每一个用户 deepcopy
            model_original = list(self.model.parameters())

            # Evaluate model each interation
            print("Evaluate global model")
            globaltestacc, globaltrainacc, globaltrainaccloss, global_test_mean_benign_acc, global_test_mean_mal_acc = self.evaluate()  # 分发了模型，当前用户的模型是全局模型， 输出所有本地数据在全局模型上测试的准确率
            globaltestasr, globaltrainasr, globaltrainasrloss, global_test_mean_benign_asr, global_test_mean_mal_asr = self.poison_evaluate(
                trigger=optimized_trigger_list, pattern=pattern)  # 输出所有本地数据在全局模型上测试的准确率

            csv_record.global_test_result.append(
                ['global', glob_iter, 'False', globaltestacc, globaltrainacc, 0,
                 global_test_mean_benign_acc, global_test_mean_mal_acc])

            csv_record.global_posiontest_result.append(
                ['global', glob_iter, 'True', globaltestasr, globaltrainasr, 0,
                 global_test_mean_benign_asr, global_test_mean_mal_asr])

            print("")

            # Evaluate gloal model on user for each interation
            print("Evaluate global model with a few step updates, which is personalized model")
            # 本地finetune一次，在本地模型上再测试   --个性化模型准确率
            if glob_iter <= attack_start:
                pertestacc, pertrainacc, pertrainloss, poiperasr, poipertrainasr, poiperloss, per_mean_ben_asr, \
                per_mean_mal_asr, per_mean_ben_acc, per_mean_mal_acc = self.evaluate_one_step(
                    per_epoch, trigger=optimized_trigger_list, pattern=pattern)
            else:
                pertestacc, pertrainacc, pertrainloss, poiperasr, poipertrainasr, poiperloss, per_mean_ben_asr, \
                per_mean_mal_asr, per_mean_ben_acc, per_mean_mal_acc = self.evaluate_one_step_poison(
                    per_epoch, trigger=optimized_trigger_list, pattern=pattern)

            csv_record.per_test_result.append(
                ['person', glob_iter, 'False', pertestacc, pertrainacc, pertrainloss, per_mean_ben_acc,
                 per_mean_mal_acc])

            csv_record.per_posiontest_result.append(
                ['person', glob_iter, 'True', poiperasr, poipertrainasr, poiperloss, per_mean_ben_asr,
                 per_mean_mal_asr])

            self.selected_users = self.select_users(glob_iter, self.num_users)  # 选择聚合用户数

            if self.attack_method == "attackall":
                for user in self.selected_users:
                    if user.id in self.malclient:
                        if glob_iter >= attack_start:
                            # 如果有恶意用户，并且该轮发起攻击，那么先进行trigger的优化
                            if self.dataset == "Mnist" or self.dataset == "FashionMnist":
                                print("optimizing the backdoor trigger !!!")
                                if First_Attack == False:
                                    temp_trigger = self.trigger_evasion_mnist(trigger=optimized_trigger_list,
                                                                              model=self.model,
                                                                              glob_iter=glob_iter,
                                                                              attackstart=attack_start)
                                    First_Attack = True
                                    optimized_trigger_list = copy.deepcopy(temp_trigger)

                                temp_trigger = self.trigger_all_mnist_l2(trigger=optimized_trigger_list,
                                                                         glob_iter=glob_iter,
                                                                         attackstart=attack_start,
                                                                         intinal_trigger=original_trigger_list,
                                                                         pattern=pattern)

                                optimized_trigger_list = copy.deepcopy(temp_trigger)

                            break

            for i in range(len(optimized_trigger_list)):
                optimized_trigger_list[i].requires_grad = False

            for user in self.selected_users:
                print("=" * 80)
                print(user.id)
                if user.id in self.malclient:
                    print("client is malicious client!")
                    # 恶意用户
                    if glob_iter >= attack_start and oneshot == 0:
                        # 可以开始投毒
                        print("conduct backdoor attack!")

                        user.poison_all_train(poison_ratio=self.poisonratio, poison_label=self.poisonlabel,
                                                  trigger=optimized_trigger_list, pattern=pattern, oneshot=0,
                                                  clip_rate=0)
                        #计算与全局模型 的距离
                        print("the distance:{}".format(self.model_dist_norm(user=user)))
                        # 测试ACC
                        acc_count, sumcount = user.test()
                        accuracy = acc_count / sumcount
                        print('ACC :{}'.format(accuracy))
                        csv_record.test_local_result.append([user.id, glob_iter, 'True', accuracy, acc_count, sumcount])

                        # 测试所有样本嵌入trigger
                        poiosnacc_count, poiosnsumcount = user.poisontest(trigger=optimized_trigger_list,
                                                                          poiosnlabel=self.poisonlabel, pattern=pattern)
                        poiosnaccuracy = poiosnacc_count / poiosnsumcount

                        print('ASR :{}'.format(poiosnaccuracy))
                        csv_record.posiontest_local_result.append(
                            [user.id, glob_iter, 'True', poiosnaccuracy, poiosnacc_count, poiosnsumcount])

                    elif glob_iter >= attack_start and First_Attack == False and oneshot == 1:
                        print("conduct modelre backdoor attack!")
                        user.poison_all_train(poison_ratio=self.poisonratio, poison_label=self.poisonlabel,
                                              trigger=optimized_trigger_list, pattern=pattern, oneshot=oneshot,
                                              clip_rate=clip_rate)
                        First_Attack = True

                        #计算与全局模型 的距离
                        print("the distance:{}".format(self.model_dist_norm(user=user)))

                        # 测试ACC
                        acc_count, sumcount = user.test()
                        accuracy = acc_count / sumcount
                        print('ACC :{}'.format(accuracy))
                        csv_record.test_local_result.append([user.id, glob_iter, 'True', accuracy, acc_count, sumcount])

                        # 测试所有样本嵌入trigger
                        poiosnacc_count, poiosnsumcount = user.poisontest(trigger=optimized_trigger_list,
                                                                          poiosnlabel=self.poisonlabel, pattern=pattern)
                        poiosnaccuracy = poiosnacc_count / poiosnsumcount

                        print('ASR :{}'.format(poiosnaccuracy))
                        csv_record.posiontest_local_result.append(
                            [user.id, glob_iter, 'True', poiosnaccuracy, poiosnacc_count, poiosnsumcount])

                    else:
                        user.train()

                        # 测试ACC
                        acc_count, sumcount = user.test()
                        accuracy = acc_count / sumcount
                        print('ACC :{}'.format(accuracy))
                        csv_record.test_local_result.append([user.id, glob_iter, 'True', accuracy, acc_count, sumcount])

                        # 测试所有样本嵌入trigger
                        poiosnacc_count, poiosnsumcount = user.poisontest(trigger=optimized_trigger_list,
                                                                          poiosnlabel=self.poisonlabel, pattern=pattern)
                        poiosnaccuracy = poiosnacc_count / poiosnsumcount

                        print('ASR :{}'.format(poiosnaccuracy))
                        csv_record.posiontest_local_result.append(
                            [user.id, glob_iter, 'True', poiosnaccuracy, poiosnacc_count, poiosnsumcount])


                else:
                    user.train()  # * user.train_samples  #训练本地模型

                    # 测试ACC
                    acc_count, sumcount = user.test()
                    accuracy = acc_count / sumcount
                    print('ACC :{}'.format(accuracy))
                    csv_record.test_local_result.append([user.id, glob_iter, 'True', accuracy, acc_count, sumcount])

                    # 测试所有样本嵌入trigger
                    poiosnacc_count, poiosnsumcount = user.poisontest(trigger=optimized_trigger_list,
                                                                      poiosnlabel=self.poisonlabel, pattern=pattern)
                    poiosnaccuracy = poiosnacc_count / poiosnsumcount

                    print('ASR :{}'.format(poiosnaccuracy))
                    csv_record.posiontest_local_result.append(
                        [user.id, glob_iter, 'True', poiosnaccuracy, poiosnacc_count, poiosnsumcount])

            # self.aggregate_parameters()  # 聚合模型
            if self.defense == "none":
                self.aggregate_parameters()
            elif self.defense == 'mkrum':
                self.Multi_Krum()
            elif self.defense == 'trim':
                self.Trimmed_Mean()
            else:
                raise ValueError("defense name wrong!")

            csv_record.save_result_csv(self.folder_path)

        self.save_results()
        self.save_poison_results()
        self.save_model()  # 保存训练完的全局模型

        return optimized_trigger_list
