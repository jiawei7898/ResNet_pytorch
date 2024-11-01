# predict.py
# ResNet —— CNN经典网络模型（PyTorch实现）
# https://blog.csdn.net/weixin_44023658/article/details/105843701

import os
import torch
from models.model import resnet34  # 从models/model.py导入resnet34模型
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import numpy as np


def main():
    # 设置环境变量以避免OpenMP重复初始化
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # 定义图像预处理步骤
    data_transform = transforms.Compose([
        transforms.Resize(256),  # 将图像调整为256x256
        transforms.CenterCrop(224),  # 中心裁剪图像至224x224
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化图像
    ])

    # 获取当前脚本的绝对路径并提取目录部分
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)

    # 加载图像
    img_path = os.path.join(current_dir, "flower4.jpg")
    if not os.path.exists(img_path):
        print(f"图像文件不存在: {img_path}")
        return
    print(f"加载图像: {img_path}")
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title("Original Image")  # 使用英文文本
    plt.show()

    # 应用图像预处理
    img = data_transform(img)
    print(f"预处理后的图像形状: {img.shape}")

    # 扩展批次维度
    img = torch.unsqueeze(img, dim=0)
    print(f"扩展批次维度后的图像形状: {img.shape}")

    # 读取类别索引文件
    class_indices_path = os.path.join(project_dir, 'data', 'class_indices.json')
    if not os.path.exists(class_indices_path):
        print(f"类别索引文件不存在: {class_indices_path}")
        return
    try:
        with open(class_indices_path, 'r') as json_file:
            class_indict = json.load(json_file)
            print("类别索引文件加载成功")
    except Exception as e:
        print(f"读取类别索引文件时出错: {e}")
        return

    # 创建模型
    model = resnet34(num_classes=5)
    print("ResNet-34模型创建成功")

    # 加载模型权重
    model_weight_path = os.path.join(project_dir, 'models', 'resNet34.pth')
    if not os.path.exists(model_weight_path):
        print(f"模型权重文件不存在: {model_weight_path}")
        return
    try:
        model.load_state_dict(torch.load(model_weight_path))
        print(f"模型权重加载成功: {model_weight_path}")
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        return

    # 将模型设置为评估模式
    model.eval()
    print("模型已设置为评估模式")

    # 预测类别
    with torch.no_grad():  # 禁用梯度计算
        output = torch.squeeze(model(img))
        print(f"模型输出: {output}")

        predict = torch.softmax(output, dim=0)
        print(f"预测概率分布: {predict}")

        predict_cla = torch.argmax(predict).numpy()
        print(f"预测类别索引: {predict_cla}")

    # 输出预测结果
    predicted_class = class_indict[str(predict_cla)]
    predicted_probability = predict[predict_cla].numpy()
    print(f"预测类别: {predicted_class}, 预测概率: {predicted_probability:.4f}")

    # 显示预测结果
    plt.figure()
    img_np = img.squeeze().permute(1, 2, 0).numpy()  # 将张量转换为numpy数组
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255  # 反归一化
    img_np = img_np.astype(np.uint8)  # 确保图像格式正确
    plt.imshow(img_np)
    plt.title(f"Predicted class: {predicted_class}, Probability: {predicted_probability:.4f}")  # 使用英文文本
    plt.show()


if __name__ == '__main__':
    main()
