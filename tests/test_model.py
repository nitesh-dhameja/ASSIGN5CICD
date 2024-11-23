import torch
import pytest
from model.model import MNISTModel
from torchvision import datasets, transforms
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = MNISTModel()
    num_params = count_parameters(model)
    print(f"\nModel has {num_params:,} parameters")
    assert num_params < 25000, "Model has too many parameters"

def test_input_output_shape():
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape is incorrect"
    print(f"\nModel output shape test passed: {output.shape}")

def test_model_accuracy():
    device = torch.device("cpu")
    model = MNISTModel().to(device)
    
    # Load saved model
    model.load_state_dict(torch.load('saved_models/mnist_model.pth', map_location=device))
    model.eval()
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    correct = 0
    total = 0
    
    print("\nEvaluating model accuracy...")
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    assert accuracy > 80, f"Model accuracy ({accuracy:.2f}%) is below 80%" 