import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy


class User:
    """
    Base class for users in federated learning.
    """

    def __init__(self, device, id, train_data, test_data, model, dataset, batch_size=0, learning_rate=0, beta=0,
                 lamda=0,
                 local_epochs=0):

        self.device = device
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.lamda = lamda
        self.local_epochs = local_epochs
        self.train_data=train_data
        self.test_dataset = test_data
        self.trainloader = DataLoader(train_data, self.batch_size)
        self.testloader = DataLoader(test_data, self.batch_size)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        self.poison_trainloader = DataLoader(train_data, self.train_samples)  # 测试投毒asr
        self.poison_testloader = DataLoader(test_data, self.test_samples)  # 成测试投毒asr
        self.dataset = dataset

        # those parameters are for persionalized federated learing.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))

    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def poison_test_dataset(self):
        test_classes = {}
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_no_id = list(range(0, len(self.test_dataset)))

    def test(self):
        # 在自己的test上测试
        self.model.eval()
        test_acc = 0
        sample = 0

        for batch_id, (datas, labels) in enumerate(self.testloader):
            x, y = datas.to(self.device), labels.to(self.device)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            sample += y.shape[0]

        return test_acc, sample

    def poisontest(self, poiosnlabel, trigger, pattern):
        # 在自己的test上测试
        self.model.eval()
        test_acc = 0
        sample = 0
        if self.dataset == "Mnist" or self.dataset == "FashionMnist":
            patterntensor = torch.ones((1, 28, 28)).float().to(self.device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[0][pos[0]][pos[1]] = 0

        elif self.dataset == "Cifar10" or self.dataset == "Cifar100":
            patterntensor = torch.ones((3, 32, 32)).float().to(self.device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[0][pos[0]][pos[1]] = 0
                patterntensor[1][pos[0]][pos[1]] = 0
                patterntensor[2][pos[0]][pos[1]] = 0

        elif self.dataset == "IoT":
            patterntensor = torch.ones((115)).float().to(self.device)
            for i in pattern:
                patterntensor[i] = 0

        # trigger is a list
        for batch_id, (datas, labels) in enumerate(self.testloader):
            x, y = datas.to(self.device), labels.to(self.device)
            new_images = x
            new_targets = y
            for index in range(0, len(x)):
                new_images[index] = self.add_pixel_pattern(x[index], trigger[new_targets[index]], pattern,
                                                           pattern_mask=patterntensor)
                new_targets[index] = poiosnlabel

            new_images, new_targets = new_images.to(self.device), new_targets.to(self.device)
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)

            output = self.model(new_images)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == new_targets)).item()
            sample += y.shape[0]

        return test_acc, sample

    def poiosn_test_attackone(self, poiosnlabel, trigger, pattern):
        self.model.eval()
        test_acc = 0
        sample = 0
        dataset = []
        for X, Y in self.testloaderfull:
            for i in range(len(X)):
                if Y[i] == 2:
                    dataset.append((X[i], Y[i]))
                    sample += 1

        dataloaders = DataLoader(dataset, self.batch_size)

        if self.dataset == "Mnist" or self.dataset == "FashionMnist":
            patterntensor = torch.ones((1, 28, 28)).float().to(self.device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[0][pos[0]][pos[1]] = 0

        elif self.dataset == "Cifar10" or self.dataset == "Cifar100":
            patterntensor = torch.ones((3, 32, 32)).float().to(self.device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[0][pos[0]][pos[1]] = 0
                patterntensor[1][pos[0]][pos[1]] = 0
                patterntensor[2][pos[0]][pos[1]] = 0

        elif self.dataset == "IoT":
            patterntensor = torch.ones((115)).float().to(self.device)
            for i in pattern:
                patterntensor[i] = 0

        for batch_id, (datas, labels) in enumerate(dataloaders):
            x, y = datas.to(self.device), labels.to(self.device)
            new_images = x
            new_targets = y
            for index in range(0, len(x)):
                new_targets[index] = poiosnlabel
                new_images[index] = self.add_pixel_pattern(x[index], trigger, pattern, pattern_mask=patterntensor)

            new_images, new_targets = new_images.to(self.device), new_targets.to(self.device)
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)

            output = self.model(new_images)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == new_targets)).item()

        return test_acc, sample

    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        for batch_id, (datas, labels) in enumerate(self.trainloader):
            x, y = datas.to(self.device), labels.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

        return train_acc, loss, self.train_samples

    def poison_train_error_and_loss(self, poiosnlabel, trigger, pattern):
        self.model.eval()
        train_acc = 0
        loss = 0
        if self.dataset == "Mnist" or self.dataset == "FashionMnist":
            patterntensor = torch.ones((1, 28, 28)).float().to(self.device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[0][pos[0]][pos[1]] = 0

        elif self.dataset == "Cifar10" or self.dataset == "Cifar100":
            patterntensor = torch.ones((3, 32, 32)).float().to(self.device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[0][pos[0]][pos[1]] = 0
                patterntensor[1][pos[0]][pos[1]] = 0
                patterntensor[2][pos[0]][pos[1]] = 0

        elif self.dataset == "IoT":
            patterntensor = torch.ones((115)).float().to(self.device)
            for i in pattern:
                patterntensor[i] = 0


        for batch_id, (datas, labels) in enumerate(self.trainloader):
            x, y = datas.to(self.device), labels.to(self.device)
            new_images = x
            new_targets = y
            for index in range(0, len(x)):
                new_images[index] = self.add_pixel_pattern(x[index], trigger[new_targets[index]], pattern,
                                                           pattern_mask=patterntensor)
                new_targets[index] = poiosnlabel

            new_images, new_targets = new_images.to(self.device), new_targets.to(self.device)
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)

            output = self.model(new_images)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == new_targets)).item()

        return train_acc, loss, self.train_samples

    def poison_one_train_error_and_loss(self, poiosnlabel, trigger, pattern):
        self.model.eval()
        train_acc = 0
        sample = 0
        dataset = []
        for X, Y in self.trainloaderfull:
            for i in range(len(X)):
                if Y[i] == 2:
                    dataset.append((X[i], Y[i]))
                    sample += 1

        dataloaders = DataLoader(dataset, self.batch_size)
        if self.dataset == "Mnist" or self.dataset == "FashionMnist":
            patterntensor = torch.ones((1, 28, 28)).float().to(self.device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[0][pos[0]][pos[1]] = 0

        elif self.dataset == "Cifar10" or self.dataset == "Cifar100":
            patterntensor = torch.ones((3, 32, 32)).float().to(self.device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[0][pos[0]][pos[1]] = 0
                patterntensor[1][pos[0]][pos[1]] = 0
                patterntensor[2][pos[0]][pos[1]] = 0

        elif self.dataset == "IoT":
            patterntensor = torch.ones((115)).float().to(self.device)
            for i in pattern:
                patterntensor[i] = 0

        for batch_id, (datas, labels) in enumerate(dataloaders):
            x, y = datas.to(self.device), labels.to(self.device)
            new_images = x
            new_targets = y
            for index in range(0, len(x)):
                new_targets[index] = poiosnlabel
                new_images[index] = self.add_pixel_pattern(x[index], trigger, pattern, pattern_mask=patterntensor)

            new_images, new_targets = new_images.to(self.device), new_targets.to(self.device)
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)

            output = self.model(new_images)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == new_targets)).item()

        return train_acc, sample

    def test_persionalized_model(self):
        self.model.eval()
        test_acc = 0
        self.update_parameters(self.persionalized_model_bar)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            # @loss += self.loss(output, y)
            # print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            # print(self.id + ", Test Loss:", loss)
        self.update_parameters(self.local_model)
        return test_acc, y.shape[0]

    def train_error_and_loss_persionalized_model(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(self.persionalized_model_bar)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        self.update_parameters(self.local_model)
        return train_acc, loss, self.train_samples

    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))

    def get_next_poison_all_train_batch(self, poison_ratio, poison_label, noise_trigger, pattern):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)

        X.to(self.device), y.to(self.device)
        poison_count = 0
        new_xs = X
        new_ys = y
        if self.dataset == "Mnist" or self.dataset == "FashionMnist":
            patterntensor = torch.ones((1, 28, 28)).float().to(self.device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[0][pos[0]][pos[1]] = 0

        elif self.dataset == "Cifar10" or self.dataset == "Cifar100":
            patterntensor = torch.ones((3, 32, 32)).float().to(self.device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[0][pos[0]][pos[1]] = 0
                patterntensor[1][pos[0]][pos[1]] = 0
                patterntensor[2][pos[0]][pos[1]] = 0

        elif self.dataset == "IoT":
            patterntensor = torch.ones((115)).float().to(self.device)
            for i in pattern:
                patterntensor[i] = 0

        for index in range(0, len(X)):
            if poison_count < poison_ratio:
                new_xs[index] = self.add_pixel_pattern(new_xs[index], noise_trigger[new_ys[index]], pattern,
                                                       pattern_mask=patterntensor)
                new_ys[index] = poison_label
                poison_count += 1
            else:
                new_ys[index] = y[index]
                new_xs[index] = X[index]

        new_xs = new_xs.to(self.device)
        new_ys = new_ys.to(self.device)

        return new_xs, new_ys

    def get_poison_batch(self, batch, trigger, pattern, poison_ratio, poison_label):
        X, y = batch
        X.to(self.device), y.to(self.device)
        poison_count = 0
        new_xs = X
        new_ys = y
        if self.dataset == "Mnist" or self.dataset == "FashionMnist":
            patterntensor = torch.ones((1, 28, 28)).float().to(self.device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[0][pos[0]][pos[1]] = 0

        elif self.dataset == "Cifar10" or self.dataset == "Cifar100":
            patterntensor = torch.ones((3, 32, 32)).float().to(self.device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[0][pos[0]][pos[1]] = 0
                patterntensor[1][pos[0]][pos[1]] = 0
                patterntensor[2][pos[0]][pos[1]] = 0

        elif self.dataset == "IoT":
            patterntensor = torch.ones((115)).float().to(self.device)
            for i in pattern:
                patterntensor[i] = 0


        for index in range(0, len(X)):
            if poison_count < poison_ratio:
                new_xs[index] = self.add_pixel_pattern(new_xs[index], trigger[new_ys[index]], pattern,
                                                       pattern_mask=patterntensor)
                new_ys[index] = poison_label
                poison_count += 1
            else:
                new_ys[index] = y[index]
                new_xs[index] = X[index]

        new_xs = new_xs.to(self.device)
        new_ys = new_ys.to(self.device)

        return new_xs, new_ys

    def add_pixel_pattern(self, ori_image, noise_trigger, pattern, pattern_mask):
        image = copy.deepcopy(ori_image).to(self.device)
        if self.dataset == "Mnist" or self.dataset == "FashionMnist":
            image = image * pattern_mask
            image = image + noise_trigger
            image = torch.clamp(image, -1, 1)

        elif self.dataset == "Cifar10" or self.dataset == "Cifar100":
            image = image * pattern_mask
            image = image + noise_trigger
            image = torch.clamp(image, -1, 1)

        elif self.dataset == "IoT":
            image = image * pattern_mask
            image = image + noise_trigger

        return image

    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
