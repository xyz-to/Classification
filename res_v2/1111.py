import torch
import torch.nn as nn


class Resnet(nn.Module):
    """resnet"""

    def __init__(self):
        super(Resnet, self).__init__()


class MoblenetV2(nn.Module):
    """moblenet"""

    def __init__(self):
        super(MoblenetV2, self).__init__()


class Total(nn.Module):
    """合起来"""

    def __init__(self, inchannels=3, num_class=101):
        super(Total, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(inchannels, 64, 7, padding=6),
            nn.Conv2d(64, 128, 3, padding=2)
        )
        self.res = Resnet()
        self.v2 = MoblenetV2()

    def forward(self, x):
        x = self.feature(x)
        #  拆分
        x1 = x[:, 1 / 2 * len(x)]
        x2 = x[1 / 2 * len(x), :]
        out1 = self.res(x1)
        out2 = self.v2(x2)
        out = torch.cat([out1, out2])  # 拼接
        out = out + x  # 残差
        return out
