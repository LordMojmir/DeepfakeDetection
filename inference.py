from concrete.ml.deployment import load_quantized_model
import numpy as np

# Load the quantized model
quantized_model = load_quantized_model("quantized_resnet50")

# Prepare a test input (for simplicity, use dummy data similar to the input shape)
test_input = np.random.rand(1, 3, 224, 224)  # Random image with the correct shape

# Perform encrypted inference using the quantized model
encrypted_output = quantized_model.predict(test_input)

print(f"Encrypted prediction result: {encrypted_output}")