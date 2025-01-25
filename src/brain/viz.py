import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def createFigureAndAxes():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def plotNeurons(ax, brain, activation_values=None):
    positions = brain.neuron_positions.cpu().numpy()
    n_neurons = len(positions)

    # Plot regular neurons
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2], c='gray', alpha=0.3
    )

    # Input neurons
    ax.scatter(
        positions[brain.input_indices, 0],
        positions[brain.input_indices, 1],
        positions[brain.input_indices, 2],
        c='yellow',
        s=100,
        label="Input",
    )

    # Output neurons
    ax.scatter(
        positions[brain.output_indices, 0],
        positions[brain.output_indices, 1],
        positions[brain.output_indices, 2],
        c='red',
        s=100,
        label="Output",
    )

    # Activation visualization
    if activation_values is not None:
        act_vals = activation_values.numpy()
        sizes = 50 + 200 * np.abs(act_vals)
        colors = plt.cm.coolwarm((act_vals + 1) / 2)  # This returns RGBA values
        colors = colors.reshape(-1, 4)  # Ensure it's Nx4 for N neurons

        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            s=sizes,
            c=colors,
            alpha=0.7,
        )


def plotSynapses(ax, brain, show_weights=False, weight_thickness=False):
    positions = brain.neuron_positions.cpu().numpy()
    weights = brain.synapse_weights.detach().cpu().numpy()

    max_thickness = 3
    if weight_thickness:
        thicknesses = max_thickness * np.abs(weights) / np.max(np.abs(weights))
    else:
        thicknesses = np.ones_like(weights)

    for i in range(len(weights)):
        from_idx = brain.synapse_indices[0][i]
        to_idx = brain.synapse_indices[1][i]

        start = positions[from_idx]
        end = positions[to_idx]

        # Calculate arrow position (70% along the line)
        arrow_pos = start + 0.7 * (end - start)
        # Calculate direction vector for arrow
        direction = end - start
        direction = direction / np.linalg.norm(direction)

        # Different colors for positive/negative weights
        color = plt.cm.coolwarm(0.9) if weights[i] > 0 else plt.cm.coolwarm(0.1)

        # Draw line
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            alpha=0.7,
            linewidth=thicknesses[i],
        )

        # Add arrow
        ax.quiver(
            arrow_pos[0],
            arrow_pos[1],
            arrow_pos[2],
            direction[0],
            direction[1],
            direction[2],
            color=color,
            alpha=0.7,
            length=0.2,
            normalize=True,
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

    SPEED_FACTOR = 200
    REPEAT_DELAY_MS = 1000

    def frame_gen():
        while True:
            for i in range(len(brain.activation_history)):
                yield i
            # Yield the last frame multiple times to create a pause
            for _ in range(int(REPEAT_DELAY_MS / (SPEED_FACTOR))):
                yield len(brain.activation_history) - 1

    anim = FuncAnimation(
        fig,
        updateAnimation,
        frames=len(brain.activation_history),
        fargs=(ax, brain, brain.activation_history),
        interval=SPEED_FACTOR,
        repeat=True,
    )

    # Keep reference to animation
    plt.show(block=True)
    return anim
