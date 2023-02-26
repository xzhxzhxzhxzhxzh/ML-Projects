import logging
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Model_MNIST(nn.Module):
    '''Define an exsample model with standard normalized weights for MNIST'''

    def __init__(self, input_dim, output_dim, units_dim):
        super().__init__()
        np.random.seed(0)
        self.weight_1 = nn.Parameter(
            torch.tensor(np.random.standard_normal((units_dim, input_dim)),
                         dtype=torch.float32))
        self.weight_2 = nn.Parameter(
            torch.tensor(np.random.standard_normal((output_dim, units_dim)),
                         dtype=torch.float32))
        self.bias_1 = nn.Parameter(
            torch.tensor(np.zeros((units_dim)), dtype=torch.float32))
        self.bias_2 = nn.Parameter(
            torch.tensor(np.zeros((output_dim)), dtype=torch.float32))

    def forward(self, x):
        # Layer 1
        x = nn.functional.linear(x, self.weight_1, self.bias_1)
        x = torch.sigmoid(x)
        # Layer 2
        x = nn.functional.linear(x, self.weight_2, self.bias_2)
        return x