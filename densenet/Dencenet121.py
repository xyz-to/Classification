"""dencenet的121层模型实验"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
from numba import cuda


# 将图片写入numpy数组
from DenseNet import DenseNet
from ImgDataset import ImgDataset


def readfile(path, label):
    image_dir = sorted(os.listdir(path))  # 读取路径中的文件夹和文件
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)  # X存放图片，四维数组
    y = np.zeros((len(image_dir)), dtype=np.uint8)  # Y存放标签
    for i, file_name in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file_name))
        x[i, :, :] = cv2.resize(img, (128, 128))
        if label:
            y[i] = int(file_name.split('_')[0])
    if label:
        return x, y
    else:
        return x


# 将数据集转成数组
data_path = 'D:\\360downloads\\food-11'
train_x, train_y = readfile(os.path.join(data_path, 'training'), True)
print('训练集长度' + str(len(train_x)))
val_x, val_y = readfile(os.path.join(data_path, 'validation'), True)
print('测试集长度' + str(len(val_x)))

"""利用Dataset和Dataloader包装Data"""
# training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 隨機將圖片水平翻轉
    transforms.RandomRotation(15),  # 隨機旋轉圖片
    transforms.ToTensor(),  # 將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
])
# testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

batch_size = 32
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# GPU
gpu = torch.device('cuda')
cpu = torch.device('cpu')

# 训练数据集
model = DenseNet().to(gpu)  # cuda()表示使用GPU
loss = nn.CrossEntropyLoss()  # CE计算Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 更新参数方式Adam
num_epch = 40

# 开始30次训练
for i in range(num_epch):
    epch_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # 启用 Batch Normalization 和 Dropout。
    # 开始图片的卷积训练
    for j, data in enumerate(train_loader):
        optimizer.zero_grad()  # 将梯度将为零
        train_pred = model(data[0].to(gpu))  # 计算概率分布
        batch_loss = loss(train_pred, data[1].to(gpu))  # 计算Loss
        batch_loss.backward()  # 计算gradient
        optimizer.step()  # 更新参数

        train_acc += np.sum(np.argmax(train_pred.to(cpu).data.numpy(), axis=1) == data[1].numpy())  # 计算整体精准度
        train_loss += batch_loss.item()  # 计算整体损失率

    model.eval()  # 不启用 Batch Normalization 和 Dropout。
    with torch.no_grad():  # 不会被计算梯度
        for ii, data in enumerate(val_loader):
            val_pred = model(data[0].to(gpu))
            batch_loss = loss(val_pred, data[1].to(gpu))
            val_acc += np.sum(np.argmax(val_pred.to(cpu).data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
        print('[{:d}/{:d}] {:.2f} sec(s) Train Acc: {:.3f} Loss: {:.3f} | Val Acc: {:.3f} loss: {:.3f}'.format(
            i + 1,
            num_epch,
            time.time() - epch_time,
            train_acc,
            train_loss,
            val_acc,
            val_loss
            ))
