import os
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader


def readfile(path, label, length):
    image_dir = sorted(os.listdir(path))  # 读取路径中的文件夹和文件
    x = np.zeros((length, 224, 224, 3), dtype=np.uint8)  # X存放图片，四维数组
    y = np.zeros(length, dtype=np.uint8)  # Y存放标签
    num = 0
    for i, dir_name in enumerate(image_dir):
        path1 = os.path.join(path, dir_name)
        image_dir_in = sorted(os.listdir(path1))
        for j, file_name in enumerate(image_dir_in):
            img = cv2.imread(os.path.join(path1, file_name))
            x[num, :, :] = cv2.resize(img, (224, 224))
            if label:
                y[num] = i
            num += 1
    if label:
        return x, y
    else:
        return x


data_path = 'D:\\pythonspace\\data\\birds-315'
train_x, train_y = readfile(os.path.join(data_path, '1'), True, 35)
print(train_x.shape, train_y.shape)