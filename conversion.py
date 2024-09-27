from concrete.ml.quantization import QuantizedModule
import torch

# Load the trained PyTorch model weights
model.load_state_dict(torch.load("resnet50.pth"))
model.eval()


# Define a QuantizedModule in Concrete ML (simplified example)
class QuantizedResNet(torch.nn.Module):
    def __init__(self, model):
        super(QuantizedResNet, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


# Convert the PyTorch model to a Concrete ML quantized model
quant_model = QuantizedResNet(model)

# Quantize the weights (simplified, Concrete ML usually supports linear models)
quant_module = QuantizedModule(quant_model)

# Convert the model to Concrete ML format for FHE inference
quantized_model = quant_module.fhe_compile()

# Save the quantized model for encrypted inference
quant_module.save("quantized_resnet50")