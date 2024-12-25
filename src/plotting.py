import matplotlib.pyplot as plt
import numpy as np


def plotTrainingMetrics(metrics):
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 6))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, metrics.items()):
        ax.plot(values)
        ax.set_title(name)
        ax.set_xlabel("Step")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plotPredictions(x, y, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(x.cpu().numpy(), y.cpu().numpy(), label="True sin(x)")
    plt.plot(x.cpu().numpy(), y_pred.cpu().numpy(), "--", label="Network output")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualizeStructure(positions, connections, indices):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    positions_np = positions.detach().numpy()
    ax.scatter(positions_np[:, 0], positions_np[:, 1], positions_np[:, 2])

    indices_np = indices.detach().numpy()
    sample_size = min(100, indices.shape[1])
    random_indices = np.random.choice(indices.shape[1], sample_size, replace=False)

    for idx in random_indices:
        start = positions_np[indices_np[0, idx]]
        end = positions_np[indices_np[1, idx]]
        ax.plot(
            [start[0], end[0]], [start[1], end[1]], [start[2], end[2]], "r-", alpha=0.1
        )

    plt.show()


def visualizeNetwork(brain):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot neurons
    positions = brain.positions.cpu().detach().numpy()
    colors = [
        "yellow" if i < 784 else "purple" if i >= len(positions) - 10 else "blue"
        for i in range(len(positions))
    ]

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, alpha=0.6)

    # Plot a sample of connections
    indices = brain.connection_indices.cpu().detach().numpy()
    weights = brain.connection_weights.cpu().detach().numpy()

    # Sample connections, more likely to show stronger weights
    n_samples = min(1000, len(weights))
    probs = np.abs(weights) / np.abs(weights).sum()
    sample_idx = np.random.choice(len(weights), size=n_samples, p=probs)

    for idx in sample_idx:
        start = positions[indices[0, idx]]
        end = positions[indices[1, idx]]
        weight = weights[idx]
        color = "red" if weight < 0 else "green"
        # Set alpha between 0.3 and 0.7 based on weight magnitude
        alpha = 0.3 + 0.4 * min(1.0, abs(weight))
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            alpha=alpha,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(
        "3D Neural Network Structure\nYellow = input, Purple = output\nGreen = positive weights, Red = negative"
    )
    plt.show()
