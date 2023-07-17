# 导入包
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet
import os
import json
import time

# 使用GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(os.path.join("train.log"), "a") as log:
    log.write(str(device) + "\n")

# 数据预处理
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


# 获取图像数据集的路径
data_root = "D:\\Program Files\\flower_test5"  # get data root path 返回上上层目录
image_path = data_root + "\\data_set\\flower_data"  # flower data_set path

# 导入训练集并进行预处理
train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])

# 按batch_size分批次加载训练集
train_loader = torch.utils.data.DataLoader(train_dataset,  # 导入的训练集
                                           batch_size=32,  # 每批训练的样本数
                                           shuffle=True,  # 是否打乱训练集
                                           num_workers=0)  # 使用线程数，在windows下设置为0

# 导入、加载 验证集
# 导入验证集并进行预处理
validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])

# 加载验证集
validate_loader = torch.utils.data.DataLoader(validate_dataset,  # 导入的验证集
                                              batch_size=32,
                                              shuffle=True,
                                              num_workers=0)

# 存储 索引：标签 的字典
# 字典，类别：索引 {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
# 将 flower_list 中的 key 和 val 调换位置
cla_dict = dict((val, key) for key, val in flower_list.items())

# 将 cla_dict 写入 json 文件中
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# 训练过程
net = AlexNet(num_classes=5, init_weights=True)  # 实例化网络（输出类型为5，初始化权重）
net.to(device)  # 分配网络到指定的设备（GPU/CPU）训练
loss_function = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(net.parameters(), lr=0.0002, weight_decay=0.001)  # 优化器（训练参数，学习率）

save_path = './AlexNet.pth'
# 如果存在预训练的模型参数文件，加载参数
# if os.path.exists(save_path):
#     net.load_state_dict(torch.load(save_path))
#     print("Loaded model parameters from:", save_path)

best_acc = 0.0

# 定义存储训练和验证损失的列表
train_losses = []
valid_losses = []

# 定义存储训练和验证准确率的列表
valid_accuracy_list = []

num_epochs = 200

for epoch in range(num_epochs):
    # 训练阶段
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss/len(train_loader))

    # 验证阶段
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            val_loss += loss_function(outputs, val_labels.to(device)).item()
            _, predicted = outputs.max(1)
            total += val_labels.size(0)
            correct += predicted.eq(val_labels.to(device)).sum().item()
    valid_losses.append(val_loss/len(validate_loader))
    # 计算准确率
    val_accuracy = 100 * correct / total
    valid_accuracy_list.append(val_accuracy/100)

    # 保存准确率最高的模型
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(net.state_dict(), save_path)

    with open("train.log", "a") as log:
        log.write(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / len(train_loader):.4f}, "
                  f"Valid Loss: {val_loss / len(validate_loader):.4f}\n")

    # 输出训练和验证信息
    print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy: {:.2f}%'
          .format(epoch+1, num_epochs, running_loss / len(train_loader), val_loss / len(validate_loader), val_accuracy))

    # 学习率调整
    optimizer.step()

print('Finished Training')

# 可视化训练和验证损失
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制训练和验证准确率折线图
plt.plot(range(1, num_epochs+1), valid_accuracy_list, label='Valid Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
