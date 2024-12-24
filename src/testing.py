from training import *
import torch
import matplotlib.pyplot as plt
import numpy as np

def trainSinX(brain):
    x = torch.linspace(-np.pi, np.pi, 100).unsqueeze(1)
    y = torch.sin(x)

    x = (x + np.pi) / (2 * np.pi)
    y = (y + 1) / 2

    x = x.to(brain.device)
    y = y.to(brain.device)

    metrics = trainNetwork(brain, x, y)
    return metrics, x, y

def plotSinXResults(brain, x, y):
    plt.figure(figsize=(10, 5))
    plt.plot(x.cpu().numpy(), y.cpu().numpy(), label='True sin(x)')
    with torch.no_grad():
        y_pred = torch.cat([brain(x[i:i+10]) for i in range(0, len(x), 10)])
        plt.plot(x.cpu().numpy(), y_pred.cpu().numpy(), '--', label='Network output')
    plt.legend()
    plt.grid(True)
    plt.show()

def trainIdentity(brain):
    """Train on y = x first as a sanity check"""
    x = torch.linspace(0, 1, 100).unsqueeze(1)
    y = x.clone()

    x = x.to(brain.device)
    y = y.to(brain.device)

    print("\nTraining on identity function (y=x)...")
    metrics = trainNetwork(brain, x, y, n_epochs=100)  # shorter training for test
    return metrics, x, y
