import torch
from torch import nn
from tqdm import tqdm
import numpy as np


def train(model, x, y, n_epochs, lr_val, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_val)
    criterion = nn.MSELoss()
    n_samples = len(x)

    metrics = {
        "loss": [],
    }

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        indices = torch.randperm(n_samples)
        epoch_loss = 0

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_indices)

        avg_loss = epoch_loss / n_samples
        metrics["loss"].append(avg_loss)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return metrics

def trainLinear(model, n_epochs, lr_val, slope=0.3, intercept=10):
    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y = slope * x + intercept

    x = x.to(model.device)
    y = y.to(model.device)

    metrics = train(model, x, y, n_epochs, lr_val, batch_size=32)
    return metrics, x, y

def trainSinX(model, n_epochs, lr_val):
    x = torch.linspace(-np.pi*4, np.pi*4, 100).unsqueeze(1)
    y = torch.sin(x)

    x = (x + np.pi) / (2 * np.pi)
    y = (y + 1) / 2

    x = x.to(model.device)
    y = y.to(model.device)

    metrics = train(model, x, y, n_epochs, lr_val, batch_size=32)
    return metrics, x, y

def trainPolynomial(model, n_epochs, lr_val, degree=4):
    x = torch.linspace(-3, 3, 300).unsqueeze(1)
    y = x.pow(degree)
    for d in range(degree - 1, -1, -1):
        coef = torch.randn(1) * 2 - 1
        y += coef * x.pow(d)

    x = x.to(model.device)
    y = y.to(model.device)

    metrics = train(model, x, y, n_epochs, lr_val, batch_size=32)
    return metrics, x, y
