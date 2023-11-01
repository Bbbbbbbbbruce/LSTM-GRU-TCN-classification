import torch.onnx
from model import TCN,GRUModel,LSTMModel
import torchvision.models as models

# Create an instance of your TCN model (replace this with your TCN model)
# model = TCN(num_classes=2).to('cpu')
model = LSTMModel(input_size=128, hidden_size=32, num_layers=1, num_classes=2, dropout=0.3).to('cpu')
# model = GRUModel(input_size=128, hidden_size=64, num_layers=1, num_classes=2, dropout=0.3).to('cpu')
# Load pre-trained weights (if available)
model.load_state_dict(torch.load('./weights/29_model.pt'))

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor with the appropriate shape
batch_size = 1  # You can adjust this
dummy_input = torch.randn(batch_size, 1, 128).to('cpu')  # Adjust the shape to match your model's input

# Export the model to ONNX format
onnx_path = './weights/model.onnx'  # Replace with your desired output path
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

print(f"Model exported to {onnx_path}")
