import torch
from torch import nn
from tqdm import tqdm
import numpy as np


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


def trainBaselineRegression(model, x, y, task_name, n_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    metrics = {
        "loss": [],
    }

    pbar = tqdm(range(n_epochs), desc=f"Training Baseline {task_name}")
    for epoch in pbar:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        metrics["loss"].append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return metrics


def trainBaselineSinX(model, n_epochs):
    x = torch.linspace(-np.pi, np.pi, 100).unsqueeze(1)
    y = torch.sin(x)

    x = (x + np.pi) / (2 * np.pi)
    y = (y + 1) / 2

    x = x.to(model.device)
    y = y.to(model.device)

    metrics = trainBaselineRegression(model, x, y, "Sin(x)", n_epochs)
    return metrics, x, y


def countWeightsAffectingBaselineOutput(model):
    total_params = 0
    param_counts = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            count = param.numel()
            param_counts[name] = count
            total_params += count

    print("baseline:")
    print(f"{total_params} total parameters")
    for name, count in param_counts.items():
        print(f"{name}: {count}")

    return total_params
