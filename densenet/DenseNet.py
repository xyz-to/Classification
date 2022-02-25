import torch.nn as nn
from densenet import DenseBlock
from densenet import Transition



class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), num_init_features=3):
        super(DenseNet, self).__init__()
        # 第一次
        self.features = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1),
            nn.MaxPool2d(3, 2)
        )
        num_features = num_init_features
        # denceblock
        for i, value in enumerate(block_config):
            block = DenseBlock.DenseBlock(value, num_features, growth_rate)
            self.features.add_module('denceblock%d' % i, block)
            num_features += growth_rate * value
            if i != len(block_config) - 1:
                transition = Transition.Transition(num_features)
                self.features.add_module('transition%d' % i, transition)
                num_features = int(num_features * 0.5)
        # 多加了一个层
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))
        self.features.add_module("pool", nn.AvgPool2d(7, 1))
        # 分类层,最后一个Denseblock没有进行特征减少
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(381*32, 381*10),
            nn.Dropout(0.5),
            nn.Linear(3810, 100)
        )

    def forward(self, x):
        feature = self.features(x)
        feature = feature.view(feature.size(0), -1)
        print(feature.shape)
        out = self.fc(feature)
        return out
