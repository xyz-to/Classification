import torch
import torch.nn as nn
from moblenet.dwConv import dwConve


class mobleNet(nn.Module):
    def __init__(self, a=1, b=1):
        super(mobleNet, self).__init__()
        self.strus = nn.Sequential(
            nn.Conv2d(3, a*32, 3, 2, padding=1),
            nn.BatchNorm2d(a*32),
            nn.ReLU(),
            dwConve(a*32, a*64),
            dwConve(a*64, a*128, 2),
            dwConve(a*128, a*128),
            dwConve(a*128, a*256, 2),
            dwConve(a*256, a*256),
            dwConve(a*256, a*512, 2),
            dwConve(a*512, a*512),
            dwConve(a*512, a*512),
            dwConve(a*512, a*512),
            dwConve(a*512, a*512),
            dwConve(a*512, a*512),
            dwConve(a*512, a*1024, 2),
            dwConve(a*1024, a*1024),
            nn.AvgPool2d(7),
        )

        self.fc = nn.Sequential(
            nn.Linear(a*1024, 512),
            nn.Linear(512, 11),
        )

    def forward(self, x):
        out = self.strus(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
