import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#
# class MINE(nn.Module):
#     def __init__(self, data_dim=100, hidden_size=10):
#         super(MINE, self).__init__()
#         self.layers = nn.Sequential(nn.Linear(data_dim, hidden_size),
#                                     nn.ReLU(),
#                                     nn.Linear(hidden_size, hidden_size),
#                                     nn.ReLU(),
#                                     nn.Linear(hidden_size, 1))
#
#     def forward(self, x, y):
#         gw_nor_vec = []
#         gw_mal_vec = []
#
#         for ig in range(len(x)):
#             gw_nor_vec.append(x[ig].reshape((-1)))
#             gw_mal_vec.append(y[ig].reshape((-1)))
#
#         gw_nor_vec = torch.cat(gw_nor_vec, dim=0)
#         gw_mal_vec = torch.cat(gw_mal_vec, dim=0)
#
#         # tiled_x = torch.stack((gw_nor_vec, gw_nor_vec), dim=0)
#         # print(tiled_x)
#         # print(tiled_x.shape)
#         # concat_y = torch.stack((gw_mal_vec, gw_mal_vec), dim=0)
#         # print(concat_y)
#         # print(concat_y.shape)
#         # inputs = torch.cat([tiled_x, concat_y], dim=1)
#         # print(inputs)
#         # print(inputs.shape)
#         logits_x = self.layers(gw_nor_vec)
#         print(logits_x)
#         logits_y = self.layers(gw_mal_vec)
#         print(logits_y)
#
#         # pred_xy = logits[:batch_size]
#         # pred_x_y = logits[batch_size:]
#         #loss = -np.log2(np.exp(1)) * (torch.mean(logits_x) - torch.log(torch.mean(torch.exp(logits_y))))
#         loss = -(torch.mean(logits_x) - torch.log(torch.mean(torch.exp(logits_y))))
#         # loss= -(-F.softplus(-torch.mean(pred_xy)) - F.softplus(torch.mean(pred_x_y)) + math.log(4))
#         # compute loss, you'd better scale exp to bit
#         return loss
class Mine(nn.Module):
    def __init__(self, input_size=100, hidden_size=10000):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output