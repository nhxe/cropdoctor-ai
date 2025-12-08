import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("cropdoctor_final_b0.onnx", providers=["CPUExecutionProvider"])

# create dummy input (1, 3, 300, 300 if that’s your model input)
dummy_input = np.random.randn(1, 3, 300, 300).astype(np.float32)

# run inference
ort_inputs = {session.get_inputs()[0].name: dummy_input}
output = session.run(None, ort_inputs)
print("Model output shape:", output[0].shape)
