import torch
import torch.nn as nn

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(768, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 1)

    def clipped_relu(self, x):
        return torch.clamp(x, 0, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.clipped_relu(x)
        x = self.fc2(x)
        x = self.clipped_relu(x)
        x = self.fc3(x)
        return x