import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data
from PIL import Image
import os
from custom_resnet import resnet18_custom

def main():
    train_dir = 'data/Train'
    val_dir = 'data/Validation'
    test_dir = 'data/Test'

    # Define transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
    }

    # Create datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, transform=data_transforms['val']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    }

    # Create dataloaders
    dataloaders = {
        'train': data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
        'val': data.DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4),
        'test': data.DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=4),
    }

    # Get class names
    class_names = image_datasets['train'].classes

    print(f"Classes: {class_names}")

    # Load the custom ResNet-18 model
    model = resnet18_custom(weights=None)  # Set weights=None if not using pretrained weights

    # Modify the final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # Set device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")
    model = model.to(device)

    # Training:

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Training mode
            else:
                model.eval()  # Evaluation mode

            # Initialize running loss and corrects as tensors on device
            running_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            running_corrects = torch.tensor(0.0, dtype=torch.float32, device=device)

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).float()

            # Compute loss and accuracy
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects / len(image_datasets[phase])

            # Move to CPU and get scalar value
            epoch_loss_value = epoch_loss.cpu().item()
            epoch_acc_value = epoch_acc.cpu().item()

            print(f'{phase.capitalize()} Loss: {epoch_loss_value:.4f} Acc: {epoch_acc_value:.4f}')
            torch.save(model.state_dict(), f'deepfake_detection_model_{epoch}.pth')

    # Save the trained model
    torch.save(model.state_dict(), 'models/deepfake_detection_model.pth')

    # Example usage of the predict function
    image_path = 'data/Test/Fake/fake_12.jpg'
    prediction = predict(image_path, model, data_transforms['test'], device, class_names)
    print(f'Predicted class: {prediction}')

def predict(image_path, model, transform, device, class_names):
    model.eval()
    img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    img = transform(img).unsqueeze(0)  # Add batch dimension
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

    return class_names[preds[0]]

if __name__ == '__main__':
    main()