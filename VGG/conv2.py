"""两个卷积层"""
import torch.nn as nn


class conv2(nn.Sequential):
    def __init__(self, inChannels, outChannels):
        super(conv2, self).__init__()
        self.add_module('conv1', nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1))
        self.add_module('bn1', nn.BatchNorm2d(outChannels))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv2', nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1))
        self.add_module('bn2', nn.BatchNorm2d(outChannels))
        self.add_module('relu2', nn.ReLU())
