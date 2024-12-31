import torch
from torch import nn
from tqdm import tqdm
import numpy as np


class BaselineMLP(nn.Module):
    def __init__(self, device, input_size, output_size, hidden_size=100):
        super().__init__()
        self.device = device

        # Create layers separately to modify initialization
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

        # Initialize weights with same scheme as Brain
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = torch.randn_like(m.weight.data) * 0.01
                m.bias.data = torch.zeros_like(m.bias.data) * 0.1

        self.to(device)

    def forward(self, x):
        return self.net(x)


def trainBaselineMNIST(model, train_loader, n_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    metrics = {"loss": [], "accuracy": []}

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"MNIST Epoch {epoch+1}/{n_epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(model.device), target.to(model.device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100. * correct / total:.1f}%"}
            )

        metrics["loss"].append(total_loss / len(train_loader))
        metrics["accuracy"].append(100.0 * correct / total)

    return metrics


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

    print("\nLearned Parameters:")
    for name, param in model.named_parameters():
        print(f"{name}:")
        print(param.data)
    return metrics


def trainBaselineIdentity(model):
    x = torch.linspace(0, 1, 100).unsqueeze(1).to(model.device)
    y = x.clone()
    return trainBaselineRegression(model, x, y, "Identity")


def trainBaselineSinX(model, n_epochs):
    x = torch.linspace(-np.pi, np.pi, 100).unsqueeze(1)
    y = torch.sin(x)

    x = (x + np.pi) / (2 * np.pi)
    y = (y + 1) / 2

    x = x.to(model.device)
    y = y.to(model.device)

    metrics = trainBaselineRegression(model, x, y, "Sin(x)", n_epochs)
    return metrics, x, y


def testBaseline(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(model.device), target.to(model.device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    print(f"\nTest accuracy: {accuracy:.1f}%")
    return accuracy


def countWeightsAffectingBaselineOutput(model):
    total_params = 0
    param_counts = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            count = param.numel()
            param_counts[name] = count
            total_params += count

    print(f"\nBaseline Parameter counts:")
    for name, count in param_counts.items():
        print(f"{name}: {count}")
    print(f"\nTotal baseline parameters: {total_params}")

    return total_params
