import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data
from PIL import Image
from sklearn.metrics import confusion_matrix
import numpy as np
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

    # Load the trained model weights
    model.load_state_dict(torch.load('models/deepfake_detection_model.pth'))

    # Run the test evaluation
    evaluate_model(model, dataloaders['test'], device, class_names)

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{cm}")

    # Extracting TP, TN, FP, FN from confusion matrix
    tn, fp, fn, tp = cm.ravel()

    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")

    # Calculate additional metrics if needed
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

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