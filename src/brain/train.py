import torch
from torch import nn
from tqdm import tqdm
import numpy as np


def train(model, x, y, n_epochs, lr_val):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_val)
    criterion = nn.MSELoss()

    # Save initial weights for comparison
    initial_weights = model.synapse_weights.clone().detach()

    metrics = {
        "loss": [],
    }

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        epoch_loss = 0
        for i in range(len(x)):
            optimizer.zero_grad()
            x_i = x[i : i + 1]
            y_i = y[i : i + 1]
            output = model(x_i)
            loss = criterion(output, y_i)
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(x)
        metrics["loss"].append(avg_loss)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return metrics


def trainLinear(model, n_epochs, lr_val, slope=0.3, intercept=10):
    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y = slope * x + intercept

    x = x.to(model.device)
    y = y.to(model.device)

    metrics = train(model, x, y, n_epochs, lr_val)
    return metrics, x, y


def trainSinX(model, n_epochs, lr_val):
    x = torch.linspace(-np.pi, np.pi, 100).unsqueeze(1)
    y = torch.sin(x)

    x = (x + np.pi) / (2 * np.pi)
    y = (y + 1) / 2

    x = x.to(model.device)
    y = y.to(model.device)

    metrics = train(model, x, y, n_epochs, lr_val)
    return metrics, x, y


def trainPolynomial(model, n_epochs, lr_val, degree=4):
    x = torch.linspace(-3, 3, 300).unsqueeze(1)
    # Create polynomial: x^3 - 2x^2 + x - 3
    y = x.pow(degree)
    for d in range(degree - 1, -1, -1):
        coef = torch.randn(1) * 2 - 1  # Random coefficient between -1 and 1
        y += coef * x.pow(d)

    x = x.to(model.device)
    y = y.to(model.device)

    metrics = train(model, x, y, n_epochs, lr_val)
    return metrics, x, y
