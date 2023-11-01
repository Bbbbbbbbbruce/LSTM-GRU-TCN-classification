import torch
import torch.nn as nn


# 定义一个简单的一维CNN模块
class SimpleTCN(nn.Module):
    def __init__(self, in_channels):
        super(SimpleTCN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels * 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(in_channels * 8)  # 添加批归一化层
        self.conv2 = nn.Conv1d(in_channels * 8, in_channels * 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(in_channels * 16)  # 添加批归一化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        # 展平
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        return x


# 定义整体网络
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # 两个一维CNN网络
        self.cnn1 = SimpleTCN(in_channels=2)  # 使用两个通道的SimpleTCN
        self.cnn2 = SimpleTCN(in_channels=1)  # 使用一个通道的SimpleTCN
        # 线性层和其它操作
        self.fc = nn.Linear(720, 60)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.conv1 = nn.Conv1d(1, 2, kernel_size=1, stride=1)

    def forward(self, x):
        # 拆分输入的三个通道
        channel1, channel2, channel3 = x.chunk(3, dim=1)
        # 合并前两个通道
        combined_channels = torch.cat((channel1, channel2), dim=1)
        out1 = self.cnn1(combined_channels)
        out2 = self.cnn2(channel3)
        # 经过线性层和激活函数
        out_combined = torch.cat((out1, out2), dim=1)
        out_combined = self.dp(self.relu(self.fc(out_combined))).unsqueeze(1)
        out_combined = self.relu(self.conv1(out_combined))
        out_combined += combined_channels

        return out_combined


if __name__ == '__main__':
    # 指定批处理大小
    batch_size = 64

    # 创建一个随机输入张量，包括批处理维度
    input_tensor = torch.randn(batch_size, 3, 60)

    # 创建模型实例
    model = CustomNet()

    # 进行前向传播
    output = model(input_tensor)

    # 输出的形状将是(batch_size, num_classes)，其中num_classes是你的类别数量
    print(output.shape)

    # 保存模型
    torch.save(model.state_dict(), './weights/custom_net.pth')
