"""16层VGG结构"""
import torch.nn as nn

from VGG.conv2 import conv2
from VGG.conv3 import conv3


class VGG1(nn.Module):
    def __init__(self):
        super(VGG1, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('l1', conv2(3, 64))
        self.feature.add_module('m1', nn.MaxPool2d(2, 2))
        self.feature.add_module('l2', conv2(64, 128))
        self.feature.add_module('m2', nn.MaxPool2d(2, 2))
        self.feature.add_module('l3', conv3(128, 256))
        self.feature.add_module('m3', nn.MaxPool2d(2, 2))
        self.feature.add_module('l4', conv3(256, 512))
        self.feature.add_module('m4', nn.MaxPool2d(2, 2))
        self.feature.add_module('l5', conv3(512, 512))
        self.feature.add_module('m5', nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 2048, 1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(2048, 1024, 1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(1024, 100, 1),
        )

    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
