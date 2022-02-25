import torch
import torch.nn as nn


class DenseLayer(nn.Sequential):
    """denceblock的内部结构"""

    def __init__(self, num_input_features, growth_rate):
        """dencelayer的结构，一个1X1，一个3X3"""
        super(DenseLayer, self).__init__()
        self.add_module('1bn', nn.BatchNorm2d(num_input_features))
        self.add_module('1relu', nn.ReLU(inplace=True))
        self.add_module('1conv', nn.Conv2d(num_input_features, growth_rate * 4, kernel_size=1))
        self.add_module('2bn', nn.BatchNorm2d(growth_rate * 4))
        self.add_module('2relu', nn.ReLU(inplace=True))
        self.add_module('2conv2', nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, padding=1))

    def forward(self, x):
        """对特征图进行合并"""
        new_feature = super(DenseLayer, self).forward(x)
        return torch.cat([x, new_feature], 1)
