import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors


def createFigureAndAxes():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def plotNeurons(ax, brain, activation_values=None):
    positions = brain.neuron_positions.cpu().numpy()

    # Plot regular neurons
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c="gray", alpha=0.5)

    # Highlight input neurons
    ax.scatter(
        positions[brain.input_indices, 0],
        positions[brain.input_indices, 1],
        positions[brain.input_indices, 2],
        c="blue",
        s=100,
        label="Input",
    )

    # Highlight output neurons
    ax.scatter(
        positions[brain.output_indices, 0],
        positions[brain.output_indices, 1],
        positions[brain.output_indices, 2],
        c="red",
        s=100,
        label="Output",
    )

    # If we have activation values, adjust size based on activation
    if activation_values is not None:
        sizes = 50 + 200 * np.abs(activation_values.numpy())
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            s=sizes,
            c="yellow",
            alpha=0.3,
        )


def plotSynapses(ax, brain, show_weights=False, weight_thickness=False):
    positions = brain.neuron_positions.cpu().numpy()
    weights = brain.synapse_weights.detach().cpu().numpy()

    # Normalize weights for thickness
    max_thickness = 3
    if weight_thickness:
        thicknesses = max_thickness * np.abs(weights) / np.max(np.abs(weights))
    else:
        thicknesses = np.ones_like(weights)

    # Plot each synapse
    for i in range(len(weights)):
        from_idx = brain.synapse_indices[0][i]
        to_idx = brain.synapse_indices[1][i]

        start = positions[from_idx]
        end = positions[to_idx]

        # Different colors for positive/negative weights
        color = "green" if weights[i] > 0 else "red"

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            alpha=0.3,
            linewidth=thicknesses[i],
        )

        # Add weight labels if requested
        if show_weights:
            mid_point = (start + end) / 2
            ax.text(
                mid_point[0],
                mid_point[1],
                mid_point[2],
                f"{weights[i]:.2f}",
                color="black",
            )


def visualizeBrain(brain, show_weights=False, weight_thickness=False):
    fig, ax = createFigureAndAxes()
    plotNeurons(ax, brain)
    plotSynapses(ax, brain, show_weights, weight_thickness)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()


def updateAnimation(frame, ax, brain, activation_history):
    ax.clear()
    plotNeurons(ax, brain, activation_history[frame])
    plotSynapses(ax, brain, show_weights=False, weight_thickness=True)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()


def animateBrain(brain):
    fig, ax = createFigureAndAxes()

    anim = FuncAnimation(
        fig,
        updateAnimation,
        frames=len(brain.activation_history),
        fargs=(ax, brain, brain.activation_history),
        interval=500,  # 500ms between frames
        repeat=True,
    )

    plt.show()
