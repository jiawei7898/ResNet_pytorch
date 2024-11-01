# spile_data.py
# AlexNet--CNN经典网络模型详解（pytorch实现）
# ResNet——CNN经典网络模型详解(pytorch实现)
# https://blog.csdn.net/weixin_44023658/article/details/105798326

# 导入os模块，用于文件和目录操作
import os
# 导入shutil模块中的copy函数，用于复制文件
from shutil import copy
# 导入random模块，用于随机选择
import random


# 定义一个创建目录的函数
def mkfile(file):
    """
    创建目录，如果目录已存在则不创建。

    :param file: 目录路径
    """
    if not os.path.exists(file):  # 检查目录是否存在
        os.makedirs(file)  # 如果不存在则创建目录
        print(f"Directory {file} created.")  # 输出创建成功的消息
    else:
        print(f"Directory {file} already exists.")  # 如果目录已存在，输出提示信息


# 定义一个分割数据集的函数
def split_data(data_dir, train_dir, val_dir, split_rate=0.1):
    """
    将数据集分为训练集和验证集。

    :param data_dir: 原始数据集目录
    :param train_dir: 训练集目录
    :param val_dir: 验证集目录
    :param split_rate: 验证集占总数据的比例
    """
    # 获取所有类别，排除.txt文件
    flower_class = [cla for cla in os.listdir(data_dir) if ".txt" not in cla]

    # 创建训练集目录
    mkfile(train_dir)
    for cla in flower_class:
        mkfile(os.path.join(train_dir, cla))  # 为每个类别创建子目录

    # 创建验证集目录
    mkfile(val_dir)
    for cla in flower_class:
        mkfile(os.path.join(val_dir, cla))  # 为每个类别创建子目录

    # 遍历每个类别
    for cla in flower_class:
        cla_path = os.path.join(data_dir, cla)  # 获取类别的路径
        images = os.listdir(cla_path)  # 获取类别下的所有图片
        num = len(images)  # 获取图片数量

        # 随机选择验证集图片
        eval_index = random.sample(images, k=int(num * split_rate))

        # 复制图片到训练集和验证集目录
        for index, image in enumerate(images):
            image_path = os.path.join(cla_path, image)  # 获取图片的完整路径
            if image in eval_index:
                new_path = os.path.join(val_dir, cla)  # 如果图片在验证集中，复制到验证集目录
                copy(image_path, new_path)
            else:
                new_path = os.path.join(train_dir, cla)  # 否则复制到训练集目录
                copy(image_path, new_path)
            print(f"\r[Class: {cla}] Processing [{index + 1}/{num}]", end="")  # 输出进度条
        print()  # 换行

    print("Data splitting completed!")  # 输出数据分割完成的消息


# 如果脚本作为主程序运行，则执行以下代码
if __name__ == '__main__':
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    project_root = os.path.dirname(current_dir)

    # 定义数据集目录
    data_dir = os.path.join(project_root, 'data/flower_photos')
    train_dir = os.path.join(project_root, 'data/train')
    val_dir = os.path.join(project_root, 'data/val')

    # 调用函数分割数据集
    split_data(data_dir, train_dir, val_dir)
