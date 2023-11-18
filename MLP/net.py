from Jtools import *

class MLPNet(nn.Module):
    def __init__(self, input_size=3, output_size=2):
        super(MLPNet, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, output_size)
        )

    def forward(self, x):
        out = self.dense(x)
        return out