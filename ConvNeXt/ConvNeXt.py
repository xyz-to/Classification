"""
    ConvNeXt-T
"""

import torch.nn as nn


class ConvNeXtT(nn.Module):
    def __init__(self):
        super(ConvNeXtT, self).__init__()
        self.feature = nn.Sequential()
        # stem层，输入图片大小应该为224，在最后一层最后变成7
        stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=4, stride=4),
            nn.BatchNorm2d(96)
        )
        self.feature.add_module("stem", stem)
        for i in range(3):
            self.feature.add_module("res2", Block(96))
        self.feature.add_module("LN2", nn.BatchNorm2d(96))
        self.feature.add_module("down1", nn.Conv2d(96, 192, kernel_size=2, stride=2))
        # 单独的采样层，尺寸缩小一倍，通道扩大一倍
        for i in range(3):
            self.feature.add_module("res3", Block(192))
        self.feature.add_module("LN3", nn.BatchNorm2d(192))
        self.feature.add_module("down2", nn.Conv2d(192, 384, kernel_size=2, stride=2))
        for i in range(9):
            self.feature.add_module("res4", Block(384))
        self.feature.add_module("LN4", nn.BatchNorm2d(384))
        self.feature.add_module("down3", nn.Conv2d(384, 768, kernel_size=2, stride=2))
        for i in range(3):
            self.feature.add_module("res5", Block(768))
        self.feature.add_module("LN5", nn.BatchNorm2d(768))
        self.feature.add_module("avgpooling", nn.AvgPool2d(7, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(768, 384),
            nn.Linear(384, 11),
        )

    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Block(nn.Module):
    def __init__(self, channels):
        super(Block, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('dwconv', nn.Conv2d(channels, channels, padding=3, kernel_size=7, groups=channels))
        self.feature.add_module('LN', nn.BatchNorm2d(channels))
        self.feature.add_module('covn1', nn.Conv2d(channels, channels * 4, kernel_size=1))
        self.feature.add_module('GELU', nn.GELU())
        self.feature.add_module('conv2', nn.Conv2d(channels * 4, channels, kernel_size=1))

    def forward(self, x):
        out = self.feature(x)
        out = x + out
        return out
