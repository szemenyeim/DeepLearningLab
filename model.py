import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, inplanes, planes, size, stride=1):
        super(Conv, self).__init__()
        # Create convolutional and batchnorm layers

    def forward(self, x):
        # Call ReLU, batchnorm and conv

        return x

class ConvNet(nn.Module):
    def __init__(self, planes, size = 3):
        super(ConvNet,self).__init__()
        # Create some convolutional and downsampling layers

        # Create classifier (10 classes)

    def forward(self, x):
        # Call layers

        return x