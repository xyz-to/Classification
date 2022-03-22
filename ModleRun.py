import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time

from VAN import modle_van
from tools import ImgDataset
from numba import cuda
from ConvNeXt import ConvNeXt
from ConvNeXt import Convblock

# 将图片写入numpy数组
from moblenet import moblenet


def readfile(path, label):
    image_dir = sorted(os.listdir(path))  # 读取路径中的文件夹和文件
    x = np.zeros((len(image_dir), 224, 224, 3), dtype=np.uint8)  # X存放图片，四维数组
    y = np.zeros((len(image_dir)), dtype=np.uint8)  # Y存放标签
    for i, file_name in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file_name))
        x[i, :, :, :] = cv2.resize(img, (224, 224))
        if label:
            y[i] = int(file_name.split('_')[0])
    if label:
        return x, y
    else:
        return x


# 将数据集转成数组
data_path = 'D:\\pythonspace\\Classification\\data'
train_x, train_y = readfile(os.path.join(data_path, 'train'), True)
print("训练集大小：{}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(data_path, 'validation'), True)
print("验证集大小：{}".format(len(val_x)))
test_x, test_y = readfile(os.path.join(data_path, 'test'), True)
print("测试集大小：{}".format(len(val_x)))

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

batch_size = 64
train_set = ImgDataset.ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset.ImgDataset(val_x, val_y, train_transform)
test_set = ImgDataset.ImgDataset(test_x, test_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# GPU
gpu = torch.device('cuda')
cpu = torch.device('cpu')

# 训练数据集，还记得改文档名字
model = modle_van.van_tiny(11).to(gpu)  # cuda()表示使用GPU
loss = nn.CrossEntropyLoss()  # CE计算Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-8)  # 更新参数方式Adam
num_epch = 20

# 创建txt文本
txt_path = 'D:/pythonspace/Classification/train_txt/' + 'VAN' + str(int(time.time()))
run_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
file = open(txt_path + '.txt', 'w')
file.writelines('运行开始时间：' + str(run_time))
file.writelines('\r')
file.writelines('训练集训练：\r')

# 开始10次训练
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
            train_acc / len(train_x),
            train_loss / len(train_x),
            val_acc / len(val_x),
            val_loss / len(val_x)
        ))
        file.writelines(
            '[{:d}/{:d}] {:.2f} sec(s) Train Acc: {:.3f} Loss: {:.3f} | Val Acc: {:.3f} loss: {:.3f}\r'.format(
                i + 1,
                num_epch,
                time.time() - epch_time,
                train_acc / len(train_x),
                train_loss / len(train_x),
                val_acc / len(val_x),
                val_loss / len(val_x)
            ))
print("---------------------------第一次训练结束-----------------------------")

train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset.ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

file.writelines('\r')
file.writelines('训练集、验证集一起训练\r')
for epoch in range(num_epch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        # 將結果 print 出來
    print('[%d/%d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
          (epoch + 1, num_epch, time.time() - epoch_start_time, \
           train_acc / train_val_set.__len__(), train_loss / train_val_set.__len__()))
    file.writelines('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f\r' % \
                    (epoch + 1, num_epch, time.time() - epoch_start_time, \
                     train_acc / train_val_set.__len__(), train_loss / train_val_set.__len__()))

"""-------------------------第二次训练结束------------------------------"""

# 测试集部分
model.eval()  #
test_pred = 0
test_acc = 0
file.writelines('\r')
file.writelines('测试集：\r')
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model(data[0].cuda())
        test_acc += np.sum(np.argmax(test_pred.to(cpu).data.numpy(), axis=1) == data[1].numpy())
    print('测试集准确率：%3.6f' % (test_acc / test_set.__len__()))
    file.writelines('测试集准确率：%3.6f\r' % (test_acc / test_set.__len__()))
run_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
file.write('运行结束时间：' + str(run_time))
file.close()
