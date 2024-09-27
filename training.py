import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def train(model, loader, criterion, optimizer, device, epochs=1):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device (MPS or CUDA)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(loader)}")


if __name__ == '__main__':
    # Check if a GPU is available and use 'mps' for Apple Silicon, else fallback to CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the ResNet50 pre-trained model with updated 'weights' parameter
    from torchvision.models import ResNet50_Weights

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Modify the model to fit the number of classes in your dataset (e.g., 10 for CIFAR-10)
    num_classes = 10
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)  # Move the model to the selected device

    # Load CIFAR-10 dataset as an example
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize CIFAR-10 (32x32) images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading CIFAR-10 dataset...")
    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)

    # DataLoader with num_workers=0 to avoid multiprocessing issues on macOS
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training the model...")
    train(model, train_loader, criterion, optimizer, device, epochs=1)

    # Save the trained model
    torch.save(model.state_dict(), "resnet50_cifar10.pth")