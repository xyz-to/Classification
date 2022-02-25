import torch
import torch.nn as nn


class dwConve(nn.Sequential):
    def __init__(self, inchannels, outchannels, stride=1):
        super(dwConve, self).__init__()
        self.add_module('dwconv', nn.Conv2d(inchannels, inchannels, 3, stride=stride, padding=1, groups=inchannels))
        self.add_module('bn1', nn.BatchNorm2d(inchannels))
        self.add_module('relu1', nn.ReLU())
        self.add_module('1conv', nn.Conv2d(inchannels, outchannels, 1))
        self.add_module('bn2', nn.BatchNorm2d(outchannels))
        self.add_module('relu2', nn.ReLU())
