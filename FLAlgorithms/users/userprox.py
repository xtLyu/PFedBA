import math
import torch
import torch.nn as nn
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import PerturbedGradientDescent


# Implementation for Fedprox clients

class UserProx(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, dataset, batch_size, learning_rate, beta,
                 lamda,
                 local_epochs):
        super().__init__(device, numeric_id, train_data, test_data, model, dataset, batch_size, learning_rate, beta,
                         lamda,
                         local_epochs)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.trainoptimizer = PerturbedGradientDescent(
            self.model.parameters(), lr=self.learning_rate, mu=0.001)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self):
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            X, y = self.get_next_train_batch()  # 训练了所有的数据
            self.trainoptimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.trainoptimizer.step(self.local_model)

        return LOSS

    def poison_all_train(self, poison_ratio, poison_label, trigger, pattern, oneshot, clip_rate):
        LOSS = 0
        last_local_model = dict()
        for name, data in self.model.state_dict().items():
            last_local_model[name] = self.model.state_dict()[name].clone()  # 获取上一轮模型参数

        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            # 获取投毒训练集
            X, y = self.get_next_poison_all_train_batch(poison_ratio=poison_ratio, poison_label=poison_label,
                                                        noise_trigger=trigger, pattern=pattern)

            self.trainoptimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.trainoptimizer.step(self.local_model)

        if oneshot == 1:
            now_local_model = dict()
            for name, data in self.model.state_dict().items():
                now_local_model[name] = self.model.state_dict()[name].clone()  # 获取上一轮模型参数

            print("scale the local model update!")
            for key, value in self.model.state_dict().items():
                target_value = last_local_model[key]
                new_value = target_value + (value - target_value) * clip_rate
                self.model.state_dict()[key].copy_(new_value)

            squared_sum = 0
            for name, layer in self.model.named_parameters():
                squared_sum += torch.sum(torch.pow(layer.data - now_local_model[name].data, 2))
            print("scaled distance:{}".format(math.sqrt(squared_sum)))

        return LOSS


    def train_one_step(self, per_epochs):
        self.model.train()
        for _ in range(per_epochs):
            for batch_id, (datas, labels) in enumerate(self.trainloader):
                X, y = datas.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

    def train_one_step_poison(self, per_epochs, trigger, pattern, poison_label, poison_ratio):
        self.model.train()
        for _ in range(per_epochs):
            for batch_id, batch in enumerate(self.trainloader):
                X, y = self.get_poison_batch(batch=batch, trigger=trigger, pattern=pattern, poison_label=poison_label,
                                             poison_ratio=poison_ratio)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
