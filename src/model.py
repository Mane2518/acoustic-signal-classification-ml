import torch.nn as nn
import torch.nn.functional as F

class AudioClassifier(nn.Module):
    """
    A Convolutional Neural Network (CNN) model designed to classify acoustic signals
    based on Mel-spectrogram inputs.
    """
    def __init__(self, num_classes=10):
        super(AudioClassifier, self).__init__()
        
        # Layer 1: Conv -> Relu -> MaxPool
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # Layer 2: Conv -> Relu -> MaxPool
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Layer 3: Conv -> Relu -> MaxPool
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # Assuming 128x128 input Mel-spectrogram
        # After 3 pools (2x2), dimensions are: 128 -> 64 -> 32 -> 16
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten for linear layers
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
