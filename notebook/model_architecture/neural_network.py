import torch.nn as nn

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        # flatten -> linear_tanh_stack -> output
        self.flatten = nn.Flatten()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(64 * (12 + 1), 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 8),
            nn.Tanh(),
        )
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_tanh_stack(x)
        x = self.output(x)
        return x