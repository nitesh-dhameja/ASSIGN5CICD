import torch  
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(0.05)
        self.dropout2 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(12 * 7 * 7, 48)
        self.fc2 = nn.Linear(48, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 12 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x 