import matplotlib.pyplot as plt
import numpy as np

def plotTrainingMetrics(metrics):
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 6))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, metrics.items()):
        ax.plot(values)
        ax.set_title(name)
        ax.set_xlabel('Step')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def plotPredictions(x, y, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(x.cpu().numpy(), y.cpu().numpy(), label='True sin(x)')
    plt.plot(x.cpu().numpy(), y_pred.cpu().numpy(), '--', label='Network output')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualizeStructure(positions, connections, indices):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    positions_np = positions.detach().numpy()
    ax.scatter(positions_np[:, 0], positions_np[:, 1], positions_np[:, 2])

    indices_np = indices.detach().numpy()
    sample_size = min(100, indices.shape[1])
    random_indices = np.random.choice(indices.shape[1], sample_size, replace=False)

    for idx in random_indices:
        start = positions_np[indices_np[0, idx]]
        end = positions_np[indices_np[1, idx]]
        ax.plot([start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]], 'r-', alpha=0.1)

    plt.show()
