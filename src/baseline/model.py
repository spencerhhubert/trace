from torch import nn


class BaselineMLP(nn.Module):
    def __init__(self, device, input_size, output_size, hidden_size=100):
        super().__init__()
        self.device = device

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

        self.to(device)

    def forward(self, x):
        return self.net(x)
