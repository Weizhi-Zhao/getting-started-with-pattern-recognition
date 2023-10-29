from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(kernel_size=5, in_channels=1, 
                               out_channels=6, stride=1, padding=2)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(kernel_size=5, in_channels=6, 
                               out_channels=16, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84,  out_features=10)

    def forward(self, x):
        x = self.avg_pool1(F.sigmoid(self.conv1(x)))
        x = self.avg_pool2(F.sigmoid(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        # x = F.softmax(self.fc3(x), dim=-1)
        x = self.fc3(x)
        return x