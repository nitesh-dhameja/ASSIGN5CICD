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
    assert accuracy > 95, f"Model accuracy ({accuracy:.2f}%) is below 95%"

def test_model_output_range():
    """
    Test if model outputs valid probabilities after softmax
    """
    model = MNISTModel()
    model.eval()
    test_input = torch.randn(1, 1, 28, 28)
    output = torch.nn.functional.softmax(model(test_input), dim=1)
    
    # Check if outputs are valid probabilities (sum to 1 and between 0 and 1)
    assert torch.allclose(output.sum(), torch.tensor(1.0), rtol=1e-5), "Output probabilities don't sum to 1"
    assert (output >= 0).all() and (output <= 1).all(), "Output values not in valid probability range"
    print("\nModel output probability test passed")

def test_batch_processing():
    """
    Test if model can handle different batch sizes
    """
    model = MNISTModel()
    model.eval()
    
    # Test with different batch sizes
    batch_sizes = [1, 32, 64, 128]
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 1, 28, 28)
        output = model(test_input)
        assert output.shape == (batch_size, 10), f"Failed to process batch size {batch_size}"
    
    print("\nBatch processing test passed for sizes:", batch_sizes)

def test_model_gradients():
    """
    Test if model gradients are properly computed
    """
    model = MNISTModel()
    criterion = torch.nn.CrossEntropyLoss()
    
    # Forward pass
    test_input = torch.randn(1, 1, 28, 28)
    test_target = torch.tensor([5])  # Random target class
    output = model(test_input)
    loss = criterion(output, test_target)
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist and are not zero for all parameters
    has_gradients = all(param.grad is not None and param.grad.abs().sum() > 0 
                       for param in model.parameters() if param.requires_grad)
    assert has_gradients, "Model parameters are missing gradients"
    print("\nGradient computation test passed")