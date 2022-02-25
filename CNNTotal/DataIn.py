import os
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tools.ImgDataset import ImgDataset


class dataIn:
    def __init__(self, path, transforms, batch_size, length, label=True, shuffle=False):
        self.label = label
        self.path = path
        self.transforms = transforms
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length = length

    def main_function(self):
        x, y = self.readfile()
        dataset = ImgDataset(x, y, self.transforms)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        return dataloader, dataset, x, y

    def readfile(self):
        image_dir = sorted(os.listdir(self.path))  # 读取路径中的文件夹和文件
        x = np.zeros((self.length, 224, 224, 3), dtype=np.uint8)  # X存放图片，四维数组
        y = np.zeros(self.length, dtype=np.uint8)  # Y存放标签
        num = 0
        for i, dir_name in enumerate(image_dir):
            path1 = os.path.join(self.path, dir_name)
            image_dir_in = sorted(os.listdir(path1))
            for j, file_name in enumerate(image_dir_in):
                img = cv2.imread(os.path.join(path1, file_name))
                x[num, :, :, :] = cv2.resize(img, (224, 224))
                if self.label:
                    y[num] = i
                num = 1 + num
        if self.label:
            return x, y
        else:
            return x
