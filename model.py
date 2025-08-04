import torch
import torch.nn as nn

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU(inplace=False)  
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=False) 
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.relu3 = nn.ReLU(inplace=False)  
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)  
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu2(x)  
        x = self.pool(x)
        
        x = x.view(-1, 7 * 7 * 64)
        x = self.fc1(x)
        x = self.relu3(x) 
        x = self.fc2(x)
        return x