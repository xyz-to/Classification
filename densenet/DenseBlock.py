import torch.nn as nn
from densenet import DenseLayer


class DenseBlock(nn.Sequential):
    """一个block包含多个内部结构"""

    def __init__(self, num_layers, num_input_features, growth_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            # 前面所有层的输出会当作输入
            layer = DenseLayer.DenseLayer(num_input_features + i*growth_rate, growth_rate)
            self.add_module('denselayar%d' % i, layer)
