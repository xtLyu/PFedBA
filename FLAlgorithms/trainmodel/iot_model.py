import torch.nn as nn
import torch.nn.functional as F
from .simple import SimpleNet

# 全连接
class IoTNet(SimpleNet):
    def __init__(self, name=None, created_time=None):
        super(IoTNet, self).__init__(f'{name}_Simple', created_time)
        self.layer1 = nn.Linear(115,128)
        self.layer2 = nn.Linear(128,64)
        self.layer3 = nn.Linear(64,32)
        self.layer4 = nn.Linear(32,11)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        # in_features = 28 * 28
        # x = x.view(-1, in_features)
        # x = self.fc2(x)

        # normal return:
        return F.log_softmax(x, dim=1)
        


