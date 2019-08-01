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
        self.p2 = Conv(planes*2,planes*4,size,2)
        self.c3 = Conv(planes*4,planes*4,size)
        self.p3 = Conv(planes*4,planes*8,size,2)
        self.c4 = Conv(planes*8,planes*8,size)
        self.p4 = Conv(planes*8,planes*16,size,2)
        self.c5 = Conv(planes*16,planes*16,size)
        self.p5 = Conv(planes*16,planes*32,size,2)

        self.classifier = nn.Conv2d(planes*32,54,1)

    def forward(self, x):
        x = self.p1(self.c1(x))
        x = self.p2(self.c2(x))
        x = self.p3(self.c3(x))
        x = self.p4(self.c4(x))
        x = self.p5(self.c5(x))
        return torch.squeeze(self.classifier(x))