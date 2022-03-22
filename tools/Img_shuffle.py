# -- coding:utf-8 --
import os
import random
from shutil import copy2
from tqdm import tqdm

"""将数据集进行打乱，分成三部分"""


class Data_shuffle():
    def __init__(self):
        print("start")

    def getDir(self, filepath):
        pathlist = os.listdir(filepath)
        return pathlist

    # 制作五类图像总的训练集，验证集和测试集所需要的文件夹，
    # 例如训练集的文件夹中装有五个文件夹，这些文件夹分别装有一定比例的五类图像
    def mkTotalDir(self, data_path):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        dic = ['train', 'validation', 'test']
        totaldir = []
        for i in range(0, 3):
            current_path = data_path + dic[i] + '/'
            # 这个函数用来判断当前路径是否存在，如果存在则创建失败，如果不存在则可以成功创建
            isExists = os.path.exists(current_path)
            if not isExists:
                os.makedirs(current_path)
                print('successful ' + dic[i])
            else:
                print('is existed, remove and rebuild')
            totaldir.append(current_path)
        return totaldir

    # 传入的参数是五类图像原本的路径，返回的是这个路径下各类图像的名称列表和图像的类别数
    def getClassesMes(self, source_path):
        classes_name_list = self.getDir(source_path)
        classes_num = len(classes_name_list)
        return classes_name_list, classes_num

    # change_path其实就是制作好的五类图像总的训练集，验证集和测试集的路径，sourcepath和上面一个函数相同
    # 这个函数是用来建训练集，测试集，验证集下五类图像的文件夹，就是建15个文件夹，当然也可以建很多类
    def mkClassDir(self, source_path, change_path):
        classes_name_list, classes_num = self.getClassesMes(source_path)
        for i in range(0, classes_num):
            current_class_path = os.path.join(change_path, classes_name_list[i])
            isExists = os.path.exists(current_class_path)
            if not isExists:
                os.makedirs(current_class_path)
                print('successful ' + classes_name_list[i])
            else:
                print('is existed')
        print('\n')

    def divideTrainValidationTest(self, source_path, train_path, validation_path, test_path):

        # classes_name_list, classes_num = self.getClassesMes(source_path)
        # classes_name_list[i]删了可以加上
        classes_num = 1

        """调用上面的函数，在训练集验证集和测试集文件夹下建立五类图像的文件夹"""
        # self.mkClassDir(source_path, train_path)
        # self.mkClassDir(source_path, validation_path)
        # self.mkClassDir(source_path, test_path)
        """
        先将一类图像的路径拿出来，将这个路径下所有这类的图片的文件名做成一个列表，使用os.listdir函数，
        然后再将列表里面的所有图像名进行shuffle就是随机打乱，
        最后从打乱后的图像按照7:2:1分别放入训练集，验证集和测试集的图像名称列表
        """
        for i in range(classes_num):
            source_image_dir = os.listdir(source_path + '/')
            random.shuffle(source_image_dir)
            train_image_list = source_image_dir[0:int(0.7 * len(source_image_dir))]
            validation_image_list = source_image_dir[int(0.7 * len(source_image_dir)):int(0.9 * len(source_image_dir))]
            test_image_list = source_image_dir[int(0.9 * len(source_image_dir)):]
            """
            找到每一个集合列表中每一张图像的原始图像位置，然后将这张图像复制到目标的路径下，一共是五类图像
            每类图像随机被分成三个去向，使用shutil库中的copy2函数进行复制，当然也可以使用move函数，但是move
            相当于移动图像，当操作结束后，原始文件夹中的图像会都跑到目标文件夹中，如果划分不正确你想重新划分
            就需要备份，不然的话很麻烦
            """
            print('开始制作第 %d 类图像:' % (i + 1))
            # 使用tqdm函数展示进度
            for train_image in tqdm(train_image_list):
                origins_train_image_path = source_path + '/' + train_image
                new_train_image_path = train_path
                copy2(origins_train_image_path, new_train_image_path)

            for validation_image in tqdm(validation_image_list):
                origins_validation_image_path = source_path + '/' + validation_image
                new_validation_image_path = validation_path
                copy2(origins_validation_image_path, new_validation_image_path)

            for test_image in tqdm(test_image_list):
                origins_test_image_path = source_path + '/' + test_image
                new_test_image_path = test_path
                copy2(origins_test_image_path, new_test_image_path)
            print('第 %d 类图像制作完成\n' % (i + 1))


source_path = 'D:/pythonspace/data/food-11/training'
path = []
duc = Data_shuffle()
path = duc.mkTotalDir(data_path='D:/pythonspace/Classification/data/')
duc.divideTrainValidationTest(source_path, path[0], path[1], path[2])
