import torch.nn as nn
import numpy as np

train_x = np.zeros((8, 9, 10))
train_y = np.zeros(10)
print(train_y)
print(int(-2 / 10 * train_y.shape[0]))
print(train_y[-1:])
test_x = train_x[:, int(-2 / 10 * train_x.shape[1]):]
test_y = train_y[int(-2 / 10 * train_y.shape[0]):]
train_x = train_x[:, :int(8 / 10 * train_x.shape[1])]
train_y = train_y[:int(8 / 10 * train_y.shape[0])]
print(test_y)
print(train_x)
