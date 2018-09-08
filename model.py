import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, inplanes, planes, size, stride=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(inplanes,planes,size,padding=size//2,stride=stride)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ConvNet(nn.Module):
    def __init__(self, planes, size = 3):
        super(ConvNet,self).__init__()
        self.c1 = Conv(3,planes,size)
        self.p1 = Conv(planes,planes*2,size,2)
        self.c2 = Conv(planes*2,planes*2,size)
        self.p2 = Conv(planes*2,planes*4,size+2,4)
        self.c3 = Conv(planes*4,planes*4,size)
        self.p3 = Conv(planes*4,planes*8,size+2,4)

        self.classifier = nn.Conv2d(planes*8,10,1)

    def forward(self, x):
        x = self.p1(self.c1(x))
        x = self.p2(self.c2(x))
        x = self.p3(self.c3(x))
        return torch.squeeze(self.classifier(x))