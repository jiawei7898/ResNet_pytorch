# train.py
# ResNet——CNN经典网络模型详解(pytorch实现)
# https://blog.csdn.net/weixin_44023658/article/details/105843701

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from models.model import resnet34  # 从models文件夹中导入ResNet34模型
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time

# 检查CUDA是否可用，并选择合适的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义数据预处理变换
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪并缩放至224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),  # 缩放至256x256
        transforms.CenterCrop(224),  # 中心裁剪至224x224
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ])
}

# 获取当前工作目录
project_root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(project_root, "..", "data")

# 打印路径以确认
print(f"Project root: {project_root}")
print(f"Data root: {data_root}")

# 加载训练数据集
train_path = os.path.join(data_root, "train")
print(f"Train path: {train_path}")
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Train path {train_path} does not exist.")
train_dataset = datasets.ImageFolder(root=train_path, transform=data_transform["train"])
train_num = len(train_dataset)
print(f"Number of training samples: {train_num}")

# 获取类别索引映射
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# 写入 JSON 文件
json_str = json.dumps(cla_dict, indent=4)
class_indices_path = os.path.join(data_root, 'class_indices.json')
with open(class_indices_path, 'w') as json_file:
    json_file.write(json_str)
print(f"类别索引已保存到 {class_indices_path}")

# 输出类别索引映射
print(f"类别索引映射: {flower_list}")

# 混淆矩阵存放位置
confusion_matrix_dir = os.path.join(project_root, '..', 'confusion_matrix')
os.makedirs(confusion_matrix_dir, exist_ok=True)
print(f"混淆矩阵存放位置: {confusion_matrix_dir}")

# 创建本次训练的混淆矩阵文件夹
timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
cm_output_dir = os.path.join(confusion_matrix_dir, f'training_{timestamp}')
os.makedirs(cm_output_dir, exist_ok=True)
print(f"混淆矩阵文件夹已创建: {cm_output_dir}")

# 定义批量大小
batch_size = 32

# 创建训练数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
print(f"Train loader created with batch size: {batch_size}")

# 加载验证数据集
val_path = os.path.join(data_root, "val")
print(f"Validation path: {val_path}")
if not os.path.exists(val_path):
    raise FileNotFoundError(f"Validation path {val_path} does not exist.")
validate_dataset = datasets.ImageFolder(root=val_path, transform=data_transform["val"])
val_num = len(validate_dataset)
print(f"Number of validation samples: {val_num}")

# 创建验证数据加载器
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
print(f"Validation loader created with batch size: {batch_size}")

# 初始化ResNet34模型，设置输出类别数为5
net = resnet34(num_classes=5)
print("ResNet34 model initialized")

# 将模型移动到指定设备
net.to(device)
print("Model moved to device")

# 定义损失函数
loss_function = nn.CrossEntropyLoss()
print("CrossEntropyLoss defined")

# 定义优化器
optimizer = optim.Adam(net.parameters(), lr=0.0001)
print("Adam optimizer defined")

# 初始化最佳准确率
best_acc = 0.0
save_path = os.path.join(project_root, '..', 'models', 'resnet34.pth')  # 模型保存路径
print(f"Model save path: {save_path}")

# 开始训练
num_epochs = 1
total_start_time = time.time()
train_losses = []
val_accuracies = []
epoch_train_times = []

for epoch in range(num_epochs):
    # 记录每个epoch的开始时间
    epoch_start_time = time.time()

    # 训练模式
    net.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    # 遍历训练数据
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        logits = net(images)

        # 计算损失
        loss = loss_function(logits, labels)

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        # 统计损失
        running_loss += loss.item()

        # 计算训练集准确率
        _, predicted = torch.max(logits, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # 打印训练进度
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print(f"\rEpoch {epoch + 1}/{num_epochs} - Train loss: {int(rate * 100):^3.0f}%[{a}->{b}]{loss:.4f}", end="")

    # 计算平均训练损失和准确率
    avg_train_loss = running_loss / (step + 1)
    train_accuracy = train_correct / train_total
    train_losses.append(avg_train_loss)

    print(
        f"\nEpoch {epoch + 1}/{num_epochs} - Average Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    # 验证模式
    net.eval()
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []

    # 不计算梯度
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            val_images, val_labels = val_images.to(device), val_labels.to(device)

            # 前向传播
            outputs = net(val_images)

            # 获取预测结果
            _, predicted = torch.max(outputs, 1)

            # 计算验证集准确数
            val_total += val_labels.size(0)
            val_correct += (predicted == val_labels).sum().item()

            # 收集预测结果和标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(val_labels.cpu().numpy())

        # 计算验证集准确率
        val_accuracy = val_correct / val_total
        val_accuracies.append(val_accuracy)

        # 保存最佳模型
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_epoch = epoch + 1
            torch.save(net.state_dict(), save_path)
            print(f"Best model saved with validation accuracy: {best_acc:.4f}")

        # 打印验证结果
        print(f'Epoch {epoch + 1}/{num_epochs} - Validation Accuracy: {val_accuracy:.4f}')

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print(f"混淆矩阵:\n{cm}")

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cla_dict.values(), yticklabels=cla_dict.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Epoch {epoch + 1})')

    # 保存混淆矩阵图
    cm_filename = f'confusion_matrix_epoch_{epoch + 1}.png'
    cm_path = os.path.join(cm_output_dir, cm_filename)
    plt.savefig(cm_path)
    plt.close()
    print(f"混淆矩阵图已保存到 {cm_path}")

    # 记录每个epoch的结束时间
    epoch_end_time = time.time()
    epoch_train_time = epoch_end_time - epoch_start_time
    epoch_train_times.append(epoch_train_time)
    print(f"Epoch {epoch + 1} training time: {epoch_train_time:.2f}s")

# 总训练时间
total_train_time = time.time() - total_start_time
print(f'总训练时间: {total_train_time:.2f}s')

# 输出最佳准确率并保存模型
print(f"最佳验证准确率: {best_acc:.4f} (Epoch {best_epoch})")
torch.save(net.state_dict(), save_path)
print(f"最佳模型已保存到 {save_path}")

# 创建新的文件夹来存放运行产生的参数变化图
timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
output_dir = os.path.join(project_root, '..', 'models', f'train_logs_{timestamp}')
os.makedirs(output_dir, exist_ok=True)
print(f"参数变化图文件夹已创建: {output_dir}")

# 绘制训练损失图
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training Loss Over Epochs\nTotal Training Time: {total_train_time:.2f}s')
plt.legend()
plt.grid(True)
loss_plot_path = os.path.join(output_dir, 'training_loss.png')
plt.savefig(loss_plot_path)
plt.close()
print(f"训练损失图已保存到 {loss_plot_path}")

# 绘制验证准确率图
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(f'Validation Accuracy Over Epochs\nBest Validation Accuracy: {best_acc:.4f} (Epoch {best_epoch})')
plt.legend()
plt.grid(True)
accuracy_plot_path = os.path.join(output_dir, 'validation_accuracy.png')
plt.savefig(accuracy_plot_path)
plt.close()
print(f"验证准确率图已保存到 {accuracy_plot_path}")

# 绘制每轮训练时间图
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), epoch_train_times, label='Training Time per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Time (s)')
plt.title(f'Training Time per Epoch\nTotal Training Time: {total_train_time:.2f}s')
plt.legend()
plt.grid(True)
time_plot_path = os.path.join(output_dir, 'training_time_per_epoch.png')
plt.savefig(time_plot_path)
plt.close()
print(f"每轮训练时间图已保存到 {time_plot_path}")

print('训练完成')