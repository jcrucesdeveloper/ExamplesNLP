import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First 2D convolutional layer: 1 input channel, 32 output features, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # Second 2D convolutional layer: 32 input channels, 64 output features, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        # Dropout layers to prevent overfitting
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolutional layers with ReLU activation
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        # Max pooling and dropout
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Flatten and pass through fully connected layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # Output layer with log softmax
        output = F.log_softmax(x, dim=1)
        return output

if __name__ == "__main__":
    # Test the network with random data
    random_data = torch.rand((1, 1, 28, 28))
    model = Net()
    result = model(random_data)
    print("Model output shape:", result.shape)
    print("Model predictions:", result) 