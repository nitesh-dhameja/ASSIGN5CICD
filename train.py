import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.model import MNISTModel
from tqdm import tqdm
import os

def evaluate(model, device, data_loader):
    """
    Evaluate the model's accuracy on a dataset
    Returns accuracy percentage
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation for inference
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def train():
    # Set device and initialize data transforms
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
    ])
    
    # Load MNIST datasets
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Initialize model, loss function, and optimizer
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    total_batches = len(train_loader)
    pbar = tqdm(total=total_batches, desc='Training')
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device and zero gradients
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        pbar.update(1)
        
        # Evaluate and log every 100 batches
        if batch_idx % 100 == 0:
            train_accuracy = evaluate(model, device, train_loader)
            test_accuracy = evaluate(model, device, test_loader)
            pbar.set_postfix({
                'batch': f'{batch_idx}/{total_batches}',
                'loss': f'{loss.item():.4f}',
                'test_acc': f'{test_accuracy:.2f}%'
            })
            print(f'\nBatch {batch_idx}/{total_batches}:')
            print(f'Test Accuracy: {test_accuracy:.2f}%')
            print('-' * 30)
    
    pbar.close()
    
    # Final evaluation and model saving
    test_accuracy = evaluate(model, device, test_loader)
    print(f'\nFinal Test Accuracy: {test_accuracy:.2f}%')
    
    # Save the trained model
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), 'saved_models/mnist_model.pth', _use_new_zipfile_serialization=False)

if __name__ == "__main__":
    train() 