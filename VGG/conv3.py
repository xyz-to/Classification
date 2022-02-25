"""3个卷积层"""
import torch.nn as nn


class conv3(nn.Sequential):
    def __init__(self, inChannels, outChannels):
        super(conv3, self).__init__()
        self.add_module('con1', nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1))
        self.add_module('bn1', nn.BatchNorm2d(outChannels))
        self.add_module('relu1', nn.ReLU())
        self.add_module('con2', nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1))
        self.add_module('bn2', nn.BatchNorm2d(outChannels))
        self.add_module('relu2', nn.ReLU())
        self.add_module('con3', nn.Conv2d(outChannels, outChannels, kernel_size=1))

