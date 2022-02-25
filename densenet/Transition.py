import torch.nn as nn


class Transition(nn.Sequential):
    """transition层"""
    def __init__(self, num_input_feature):
        super(Transition, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(num_input_feature))
        self.add_module('conv', nn.Conv2d(num_input_feature, int(0.5*num_input_feature), kernel_size=1))
        self.add_module('pool', nn.AvgPool2d(2, stride=2))
