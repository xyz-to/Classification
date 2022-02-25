from torch import nn

def shuffle_channels(x, groups):
    """通道洗牌"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # 四维数组变成五维数组
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # 矩阵转置
    x = x.transpose(1, 2).contiguous()
    # 变成原始维度作为输出
    x = x.view(batch_size, channels, height, width)
    return x


class shufflenet1(nn.Sequential):
    """不需要步长为2的单元结构"""

    def __init__(self, inchannels, outchannels, g):
        super(shufflenet1, self).__init__()
        bottleneck_channels = outchannels // 4
        self.feature = nn.Sequential()
        self.feature.add_module('gconv1', nn.Conv2d(inchannels, bottleneck_channels, 1, groups=g))
        self.feature.add_module('bn1', nn.BatchNorm2d(bottleneck_channels))
        self.feature.add_module('relu1', nn.ReLU())
        self.feature.add_module('dwcon1', nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, groups=bottleneck_channels))
        self.feature.add_module('bn2', nn.BatchNorm2d(bottleneck_channels))
        self.feature.add_module('gconv2', nn.Conv2d(bottleneck_channels, outchannels, groups=g))
        self.feature.add_module('bn3', nn.BatchNorm2d(outchannels))

    def forward(self, x):
        out = self.feature(x)





