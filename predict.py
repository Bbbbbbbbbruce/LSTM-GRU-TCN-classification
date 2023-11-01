# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import TCN
from datas import generate_data
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Generate data
test_data, test_labels = generate_data(128, 8000, 1)

# Hyperparameters
batch_size = 64
num_classes = 7

test_dataset = TensorDataset(torch.Tensor(test_data), torch.LongTensor(test_labels))
test_loader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device('mps')
print('test data:', len(test_dataset))
print('device: ', device)
model = TCN(num_classes).to(device)
model.load_state_dict(torch.load('./weights/model.pt'))
model.eval()
# torch.save(model.state_dict(), './weights/model.pt')
# Testing
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs.permute(0, 2, 1))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print('f{i}: ', predicted == labels)

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy}")
