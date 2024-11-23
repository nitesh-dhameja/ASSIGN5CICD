import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
from torchvision import datasets
import numpy as np

def get_augmentation_transforms():
    """
    Returns the augmentation transforms pipeline
    """
    return transforms.Compose([
        transforms.RandomRotation(10),  # Rotate by up to 10 degrees
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Translate by up to 10%
            scale=(0.9, 1.1),  # Scale by 90-110%
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_test_transforms():
    """
    Returns the test transforms pipeline
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def save_augmented_samples(num_samples=5):
    """
    Saves augmented samples to visualize the augmentations
    """
    # Create directory for augmented samples
    os.makedirs('augmented_samples', exist_ok=True)
    
    # Load MNIST dataset
    dataset = datasets.MNIST('data', train=True, download=True)
    
    # Get augmentation transforms
    aug_transforms = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.ToTensor(),
    ])
    
    # Select random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx, sample_idx in enumerate(indices):
        image, label = dataset[sample_idx]
        
        # Save original image
        original_tensor = transforms.ToTensor()(image)
        save_image(original_tensor, f'augmented_samples/original_{idx}_label_{label}.png')
        
        # Save 3 different augmentations for each image
        for aug_idx in range(3):
            augmented = aug_transforms(image)
            save_image(augmented, f'augmented_samples/augmented_{idx}_version_{aug_idx}_label_{label}.png')

if __name__ == "__main__":
    save_augmented_samples() 