# model.py
import torch
import torch.nn as nn

import torch.nn as nn


class TCN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super(TCN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)  # Batch Normalization after conv1
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)  # Batch Normalization after conv2
        self.pool2 = nn.AvgPool1d(kernel_size=4)
        self.fc1 = nn.Linear(512, 64)
        self.dp = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.classes = nn.Linear(64, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        re = x
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + re
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.dp(self.relu(self.fc1(x)))
        x = self.classes(x)
        return x


class GRUModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=64, num_layers=1, num_classes=2, dropout=0.4):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Forward pass through the GRU layers
        out, _ = self.gru(x)

        # Select the output from the last time step (many-to-one architecture)
        out = out[:, -1, :]

        # Apply dropout and pass through the fully connected layer
        out = self.dropout(out)
        out = self.fc(out)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=32, num_layers=1, num_classes=2, dropout=0.4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Forward pass through the LSTM layers
        out, _ = self.lstm(x)

        # Select the output from the last time step (many-to-one architecture)
        out = out[:, -1, :]

        # Apply dropout and pass through the fully connected layer
        out = self.dropout(out)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # 指定批处理大小
    batch_size = 64

    # 创建模型实例
    model = TCN(num_classes=2).to(device)
    # Create the GRU model instance
    # model = GRUModel(input_size=256, hidden_size=64, num_layers=1, num_classes=2, dropout=0.3).to(device)
    # model = LSTMModel(input_size=128, hidden_size=32, num_layers=1, num_classes=2, dropout=0.3).to(device)

    # 创建一个随机输入张量，包括批处理维度
    input_tensor = torch.randn(batch_size, 1, 128).to(device)

    # 使用模型进行前向传递
    output = model(input_tensor)

    # 输出的形状将是(batch_size, num_classes)，其中num_classes是你的类别数量
    print(output.shape)
