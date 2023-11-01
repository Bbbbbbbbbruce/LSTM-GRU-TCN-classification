import onnxruntime
from to_onnx import onnx_path, dummy_input

# 加载ONNX模型并预测
ort_session = onnxruntime.InferenceSession(onnx_path)
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
ort_outputs = ort_session.run(None, ort_inputs)
print(ort_outputs[0])
