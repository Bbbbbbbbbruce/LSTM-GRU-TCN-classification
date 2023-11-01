import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import TCN
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# 数据目录
data_dir = './data/txt_fre_data'

# 获取子文件夹名称作为标签
labels = [label for label in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, label))]
print(labels)
data = []
data_labels = []
# folder_path = './data/txt_fre_data/0'
for i in labels:
    folder_path = data_dir + '/' + i
    # 获取文件夹中所有txt文件的名称
    txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
    for j in txt_files:
        txt_path = folder_path + '/' + j
        # 打开文件并按行读取内容
        with open(txt_path, 'r') as file:
            lines = file.readlines()
        # 遍历每一行数据
        for line in lines:
            # 使用逗号分隔每行数据，并去除末尾的换行符
            row = line.strip().split(',')
            # 将每行数据添加到结果列表
            data.append(row)
            data_labels.append(int(i))

# 创建自定义数据集实例
data, data_labels = np.array(data), np.array(data_labels)

# 划分训练集和验证集
val_ratio = 0.15
# 创建 DataLoader 加载数据
batch_size = 64  # 设置每个批次的大小
num_classes = 2
epochs = 30
learning_rate = 0.001

# 将数据和标签合并在一起
combined_data = np.column_stack((data, data_labels))

# 随机打乱数据
np.random.shuffle(combined_data)

# 计算验证集的大小
validation_ratio = 0.15
num_samples = combined_data.shape[0]
num_validation_samples = int(num_samples * validation_ratio)

# 分割成训练集和验证集
val_data = combined_data[:num_validation_samples, :-1].astype(np.float32)  # 去掉最后一列，即标签列
val_labels = combined_data[:num_validation_samples, -1].astype(np.float32)  # 只取最后一列，即标签列
train_data = combined_data[num_validation_samples:, :-1].astype(np.float32)
train_labels = combined_data[num_validation_samples:, -1].astype(np.float32)

# 将NumPy数组转换为PyTorch张量
train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
val_data_tensor = torch.tensor(val_data, dtype=torch.float32)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

# Hyperparameters
batch_size = 64
num_classes = 2
epochs = 30
learning_rate = 0.001

# Create DataLoaders
train_dataset = TensorDataset(torch.Tensor(train_data), torch.LongTensor(train_labels))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(torch.Tensor(val_data), torch.LongTensor(val_labels))
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Create model, loss function and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TCN(num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
print('train:', len(train_dataset))
print('val:', len(val_dataset))

print(device)

# Lists to store training and validation statistics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training and validation loop
for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        inputs = inputs.view(inputs.size(0), -1, inputs.size(1))
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = correct / total

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        for inputs, labels in val_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1, inputs.size(1))
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = correct / total
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1}/{epochs}, Train Acc: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')


    # Save the trained model
    torch.save(model.state_dict(), './weights/{}_model.pt'.format(epoch))

# Plot training and validation statistics
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('./weights/save/training_stats.png')
plt.show()
