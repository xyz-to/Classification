import torch.nn as nn
import torch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )


class MobileNetV2(nn.Module):
    def __init__(self, in_channel, stride):
        super(MobileNetV2, self).__init__()
        expand_ratio = 6
        hidden_channel = in_channel * expand_ratio,
        layer = []
        layer.extend(
            [
                ConvBNReLU(in_channel, hidden_channel, kernel_size=1),
                ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
                ConvBNReLU(hidden_channel, in_channel, kernel_size=1),
                nn.BatchNorm2d(in_channel),
                ConvBNReLU(in_channel, hidden_channel, kernel_size=1),
                ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
                ConvBNReLU(hidden_channel, in_channel, kernel_size=1),
                nn.BatchNorm2d(in_channel),
                nn.MaxPool2d()
            ])
        self.mobv2 = nn.Sequential(*layer)

    def forward(self, x):
        return x + self.mobv2(x)


class ToTal(nn.Module):
    def __init__(self, num_classes=1000):
        super(ToTal, self).__init__()
        self.mobv2 = MobileNetV2(128, 1)
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.feature(x)
        x = self.mobv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
