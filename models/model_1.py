import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class Model_1(nn.Module):
    '''Define an exsample CNN model.
    
    Ref: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    '''

    def __init__(self):
        super().__init__()
        # Convolutional layer
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x