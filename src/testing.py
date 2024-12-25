from training import *
import torch
import matplotlib.pyplot as plt
import numpy as np


def trainSinX(brain):
    # More points in training set
    x = torch.linspace(-2 * np.pi, 2 * np.pi, 200).unsqueeze(1)
    y = torch.sin(x)

    # Normalize to [0,1] range
    x = (x + 2 * np.pi) / (4 * np.pi)
    y = (y + 1) / 2

    x = x.to(brain.device)
    y = y.to(brain.device)

    metrics = trainNetwork(brain, x, y, n_epochs=200)

    # Test on wider range after training
    with torch.no_grad():
        x_test = torch.linspace(-4 * np.pi, 4 * np.pi, 400).unsqueeze(1)
        y_test = torch.sin(x_test)

        # Normalize test data the same way
        x_test_norm = (x_test + 2 * np.pi) / (4 * np.pi)
        x_test_norm = x_test_norm.to(brain.device)

        # Get predictions in batches to avoid memory issues
        y_pred = []
        for i in range(0, len(x_test_norm), 10):
            batch = x_test_norm[i : i + 10]
            pred = brain(batch)
            y_pred.append(pred)
        y_pred = torch.cat(y_pred)

        # Denormalize predictions
        y_pred = y_pred * 2 - 1

        plt.figure(figsize=(15, 5))
        plt.plot(x_test.cpu().numpy(), y_test.cpu().numpy(), label="True sin(x)")
        plt.plot(
            x_test.cpu().numpy(), y_pred.cpu().numpy(), "--", label="Network output"
        )
        plt.axvspan(
            -2 * np.pi, 2 * np.pi, color="gray", alpha=0.1, label="Training range"
        )
        plt.legend()
        plt.grid(True)
        plt.title("Sin(x) Prediction vs Ground Truth")
        plt.show()

    return metrics


def plotSinXResults(brain, x, y):
    plt.figure(figsize=(10, 5))
    plt.plot(x.cpu().numpy(), y.cpu().numpy(), label="True sin(x)")
    with torch.no_grad():
        y_pred = torch.cat([brain(x[i : i + 10]) for i in range(0, len(x), 10)])
        plt.plot(x.cpu().numpy(), y_pred.cpu().numpy(), "--", label="Network output")
    plt.legend()
    plt.grid(True)
    plt.show()


def trainIdentity(brain):
    x = torch.linspace(0, 1, 100).unsqueeze(1)
    y = x.clone()

    x = x.to(brain.device)
    y = y.to(brain.device)

    print("\nTraining on identity function (y=x)...")
    metrics = trainNetwork(brain, x, y, n_epochs=100)
    return metrics


def trainMNIST(brain, train_loader, n_epochs=10):
    optimizer = torch.optim.Adam(brain.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(
        brain.device
    )  # Move criterion to same device as brain

    metrics = {
        "loss": [],
        "grad_norm": [],
        "activation_mean": [],
        "activation_std": [],
        "active_neuron_ratio": [],
        "accuracy": [],
    }

    pbar = tqdm(range(n_epochs), desc="Training MNIST")
    for epoch in pbar:
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(brain.device), target.to(brain.device)
            optimizer.zero_grad()

            batch_size = data.size(0)
            data = data.view(batch_size, -1)  # flatten images

            output = brain(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            total_loss += loss.item()

            # Collect debug metrics
            with torch.no_grad():
                grad_norm = torch.norm(
                    torch.stack(
                        [
                            p.grad.norm()
                            for p in brain.parameters()
                            if p.grad is not None
                        ]
                    )
                )

                active_neurons = (brain.activations.abs() > 0.01).float().mean()
                act_mean = brain.activations.mean()
                act_std = brain.activations.std()

                metrics["grad_norm"].append(grad_norm.item())
                metrics["activation_mean"].append(act_mean.item())
                metrics["activation_std"].append(act_std.item())
                metrics["active_neuron_ratio"].append(active_neurons.item())

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        metrics["loss"].append(avg_loss)
        metrics["accuracy"].append(accuracy)

        pbar.set_postfix(
            {
                "loss": f"{avg_loss:.4f}",
                "acc": f"{accuracy:.1f}%",
                "act%": f'{metrics["active_neuron_ratio"][-1]:.2%}',
            }
        )

    return metrics
