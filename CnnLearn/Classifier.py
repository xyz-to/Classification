import torch
import torch.nn as nn


class Classifier(nn.Module):
    """CNN模型，5个卷积池化层，3个线性整合层"""

    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(  # 有序容器，计算机执行时有序的执行
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),  # 归一化
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(2, 2, 0),  # 池化层

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )

        self.fc = nn.Sequential(
            nn.Linear(25088, 12544),
            nn.ReLU(),
            nn.Linear(12544, 6272),
            nn.ReLU(),
            nn.Linear(6272, 3136),
            nn.ReLU(),
            nn.Linear(3136, 60)
        )

    def forward(self, x):  # self表示实例对象本身
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)  # 将卷积后的数据拉直
        print(out.shape)
        return self.fc(out)



