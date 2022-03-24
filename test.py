import csv

import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import cv2

# 使用第一张与第三张GPU卡
file=open('./train_txt/1.txt', "w+")
for i in range(10):
    content = "第{0}条数据".format(i)
    file.write(content + '\n')
    print(content)
