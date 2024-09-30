import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
import os

from concrete.fhe import Configuration
from concrete.ml.torch.compile import compile_torch_model

from custom_resnet import resnet18_custom  # Assuming custom_resnet.py is in the same directory

# Load class names (FLIPPED as ['Fake', 'Real'])
class_names = ['Fake', 'Real']  # Fix the incorrect mapping

# Load the trained model
def load_model(model_path, device):
    print("load_model")
    model = resnet18_custom(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))  # Assuming 2 classes: Fake and Real
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


def load_secure_model(model):
    print("Compiling secure model...")
    secure_model = compile_torch_model(
        model.to("cpu"), 
        n_bits={"model_inputs": 4, "op_inputs": 3, "op_weights": 3, "model_outputs": 5},
        rounding_threshold_bits={"n_bits": 7, "method": "APPROXIMATE"},
        p_error=0.05,
        configuration=Configuration(enable_tlu_fusing=True, print_tlu_fusing=False, use_gpu=False),
        torch_inputset=torch.rand(10, 3, 224, 224)
    )
    return secure_model

# Load models
model = load_model('models/deepfake_detection_model.pth', 'cpu')
# secure_model = load_secure_model(model)

# Image preprocessing (match with the transforms used during training)
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prediction function
def predict(image, mode, expected_output=None):
    device = 'cpu'

    # Apply transformations to the input image
    image = Image.open(image).convert('RGB')
    image = data_transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Inference
    with torch.no_grad():
        start_time = time.time()
        
        if mode == "Fast":
            # Fast mode (less computation)
            outputs = model(image)
        elif mode == "Secure":
            # Secure mode (e.g., running multiple times for higher confidence)
            detached_input = image.detach().numpy()
            outputs = torch.from_numpy(secure_model.forward(detached_input, fhe="simulate"))
        
        _, preds = torch.max(outputs, 1)
        elapsed_time = time.time() - start_time

    predicted_class = class_names[preds[0]]
    
    # Compare predicted and expected output
    expected_output_message = f"Expected: {expected_output}" if expected_output else "Expected: Not Provided"
    predicted_output_message = f"Predicted: {predicted_class}"
    
    return predicted_output_message, expected_output_message, f"Time taken: {elapsed_time:.2f} seconds"


# Path to example images for "Fake" and "Real" classes along with expected outputs
example_images = [
    ["./data/fake/fake_1.jpeg", "Fast", "Fake"],   # Fake example with expected output
    ["./data/real/real_1.jpg", "Fast", "Real"],   # Real example with expected output
]

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="filepath", label="Upload an Image"),  # Image input
        gr.Radio(choices=["Fast", "Secure"], label="Inference Mode", value="Fast"),  # Inference mode
        gr.Textbox(label="Expected Output", value=None, placeholder="Optional: Enter expected output (Fake/Real)")  # Expected output (optional)
    ],
    outputs=[
        gr.Textbox(label="Prediction"),  # Prediction output
        gr.Textbox(label="Expected Output"),  # Expected output for comparison
        gr.Textbox(label="Time Taken")  # Time taken output
    ],
    examples=[  # Include expected outputs in examples, but only image and mode will be used
        ["./data/fake/fake_1.jpeg", "Fast"],  # Fake example
        ["./data/real/real_1.jpg", "Fast"],   # Real example
    ],
    title="Deepfake Detection Model",
    description="Upload an image or select a sample and choose the inference mode (Fast or Secure)."
)

if __name__ == "__main__":
    iface.launch(share=True)