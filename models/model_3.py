import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Model_3(nn.Module):
    '''Define an exsample CNN model, which should slightly better that Model_2.
    
    Ref: https://shonit2096.medium.com/cnn-on-cifar10-data-set-using-pytorch-34be87e09844
    '''

    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            # Convolutional Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolutional Layer block 2
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Convolutional Layer block 3
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(4096, 1024),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(1024, 512),
                                      nn.ReLU(inplace=True), nn.Dropout(p=0.1),
                                      nn.Linear(512, 10))

    def forward(self, x):
        # Convolutional layers
        x = self.conv_layer(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected layer
        x = self.fc_layer(x)
        return x