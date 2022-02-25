from ModleRun import modelRun
import os
from shufflenet import shufflenetv1

if __name__ == '__main__':
    # 数据位置
    data_path = []
    path = 'D:\\pythonspace\\data\\birds-100'
    data_path.append(os.path.join(path, 'train'))
    data_path.append(os.path.join(path, 'valid'))
    data_path.append(os.path.join(path, 'test'))

    # 数据大小
    lenth = [8671, 300, 300]

    # 开始跑模型
    modle_run = modelRun(
        data_path=data_path,
        model=shufflenetv1.ShuffleNet(),
        lr=0.0001,
        epoch=10,
        batch_size=16,
        length=lenth
    )
    modle_run.main_function()
