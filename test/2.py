import os
import shutil

import numpy as np

data_path = 'D:\\pythonspace\\data\\birds-100'
train_path = os.path.join(data_path, 'train')
val_path = os.path.join(data_path, 'valid')
test_path = os.path.join(data_path, 'test')


def removeImage(path):
    """删除文件"""
    dir_name = sorted(os.listdir(path))
    remove_name = np.array(dir_name)[60:]
    for i, data in enumerate(remove_name):
        path1 = os.path.join(path, data)
        shutil.rmtree(path1)
    dir_name = os.listdir(path)
    print('去除后剩余的文件数量' + str(len(dir_name)))


def numFiles(path):
    """查看文件总数"""
    dir_name = sorted(os.listdir(path))
    lenth = 0
    for i, data in enumerate(dir_name):
        path1 = os.path.join(path, data)
        files = sorted(os.listdir(path1))
        lenth += len(files)
    print('文件总数 {}'.format(lenth))


removeImage(test_path)
removeImage(train_path)
removeImage(val_path)
numFiles(test_path)
numFiles(train_path)
numFiles(val_path)
