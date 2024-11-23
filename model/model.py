import torch  
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # First convolutional layer: input_channels=1 (grayscale), output_channels=4
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        # Second convolutional layer: input_channels=4, output_channels=8
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        
        # Dropout layers for regularization
        self.dropout1 = nn.Dropout2d(0.05)  # 5% dropout on convolutional layers
        self.dropout2 = nn.Dropout(0.1)     # 10% dropout on fully connected layer
        
        # Fully connected layers
        # After two 2x2 max pools, 28x28 becomes 7x7, with 8 channels
        self.fc1 = nn.Linear(8 * 7 * 7, 32)  # Flatten and reduce to 32 features
        self.fc2 = nn.Linear(32, 10)         # Output layer (10 digits)
        
        # Pooling and activation functions
        self.pool = nn.MaxPool2d(2, 2)       # 2x2 max pooling
        self.relu = nn.ReLU()                # ReLU activation
        
    def forward(self, x):
        # First conv block: conv -> relu -> pool -> dropout
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout1(x)
        
        # Second conv block: conv -> relu -> pool -> dropout
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout1(x)
        
        # Flatten the output for fully connected layers
        x = x.view(-1, 8 * 7 * 7)
        
        # Fully connected layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x 