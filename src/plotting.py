import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import deque
import matplotlib.animation as animation
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter


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


def analyzeConnectivity(brain):
    input_size = brain.input_size
    neuron_count = brain.neuron_count
    input_neurons = torch.arange(input_size, device=brain.device)
    output_neurons = torch.arange(
        neuron_count - input_size, neuron_count, device=brain.device
    )

    # Build adjacency list
    adjacency_list = [[] for _ in range(neuron_count)]
    from_indices = brain.connection_indices[0].cpu().numpy()
    to_indices = brain.connection_indices[1].cpu().numpy()

    for src, dst in zip(from_indices, to_indices):
        adjacency_list[src].append(dst)

    hop_counts = []

    # For each input neuron, find the shortest path to any output neuron
    for input_neuron in input_neurons.cpu().numpy():
        visited = [False] * neuron_count
        queue = deque()
        queue.append((input_neuron, 0))
        visited[input_neuron] = True
        found = False

        while queue and not found:
            current_node, hops = queue.popleft()
            if current_node in output_neurons.cpu().numpy():
                hop_counts.append(hops)
                found = True
                break
            for neighbor in adjacency_list[current_node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, hops + 1))

        if not found:
            # No path found from this input neuron to any output neuron
            hop_counts.append(float("inf"))

    # Filter out infinities (disconnected neurons)
    finite_hop_counts = [hops for hops in hop_counts if hops != float("inf")]
    if finite_hop_counts:
        min_hops = min(finite_hop_counts)
        max_hops = max(finite_hop_counts)
        avg_hops = sum(finite_hop_counts) / len(finite_hop_counts)
        print(f"Minimum hops from inputs to outputs: {min_hops}")
        print(f"Maximum hops from inputs to outputs: {max_hops}")
        print(f"Average hops from inputs to outputs: {avg_hops:.2f}")
    else:
        print("No paths exist from any input neuron to any output neuron.")


def countWeightsAffectingOutputBasedOnInput(brain):
    neuron_count = brain.neuron_count
    input_neurons = brain.input_neurons.cpu().numpy()
    output_neurons = set(brain.output_neurons.cpu().numpy())

    # Build adjacency list
    adjacency_list = [[] for _ in range(neuron_count)]
    from_indices = brain.connection_indices[0].cpu().numpy()
    to_indices = brain.connection_indices[1].cpu().numpy()

    for idx, (src, dst) in enumerate(zip(from_indices, to_indices)):
        adjacency_list[src].append((dst, idx))

    visited_neurons = set(input_neurons)
    queue = deque(input_neurons)
    visited_weights = set()

    while queue:
        current_neuron = queue.popleft()
        for neighbor, weight_idx in adjacency_list[current_neuron]:
            # Collect all weights traversed, regardless of whether neighbor was visited
            visited_weights.add(weight_idx)
            if neighbor not in visited_neurons:
                visited_neurons.add(neighbor)
                queue.append(neighbor)

    # Check if output neurons are reachable
    neurons_affecting_output = visited_neurons.intersection(output_neurons)
    if neurons_affecting_output:
        num_weights_affecting_output = len(visited_weights)
        print(
            f"Number of weights affecting output neurons based on inputs: {num_weights_affecting_output}"
        )
    else:
        print("No output neurons are reachable from the input neurons.")
        num_weights_affecting_output = 0

    return num_weights_affecting_output


def plotFunctionResults(x, y_true, y_pred, title="Function Comparison"):
    plt.figure(figsize=(10, 5))
    plt.plot(x.cpu().numpy(), y_true.cpu().numpy(), label="True function")
    plt.plot(x.cpu().numpy(), y_pred.cpu().numpy(), "--", label="Model output")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()


def animateNetworkActivity(brain):
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d.art3d import Line3D
    import numpy as np

    positions = brain.positions.cpu().detach().numpy()
    activations_over_time = brain.activations_over_time

    if torch.is_tensor(activations_over_time):
        activations_over_time = activations_over_time.cpu().numpy()

    time_steps = activations_over_time.shape[0]

    connections_from = brain.connection_indices[0].cpu().numpy()
    connections_to = brain.connection_indices[1].cpu().numpy()
    weights = brain.connection_weights.detach().cpu().numpy()

    total_neurons = len(positions)
    total_synapses = len(weights)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    max_activation = np.max(np.abs(activations_over_time))
    norm = plt.Normalize(-max_activation, max_activation)
    weight_norm = plt.Normalize(-np.max(np.abs(weights)), np.max(np.abs(weights)))
    neuron_cmap = plt.get_cmap("coolwarm")
    connection_cmap = plt.get_cmap("RdYlBu")

    # Plot connections (synapses) with arrows
    for i, (from_idx, to_idx, weight) in enumerate(
        zip(connections_from, connections_to, weights)
    ):
        # Get positions for start and end points
        start = positions[from_idx]
        end = positions[to_idx]

        # Calculate direction vector
        direction = end - start

        # Create points for the line (excluding the very end to leave room for arrow)
        arrow_length = (
            np.linalg.norm(direction) * 0.2
        )  # Arrow length is 20% of total length
        direction_normalized = direction / np.linalg.norm(direction)
        arrow_start = end - direction_normalized * arrow_length

        # Draw main connection line
        connection_color = connection_cmap(weight_norm(weight))
        line = Line3D(
            [start[0], arrow_start[0]],
            [start[1], arrow_start[1]],
            [start[2], arrow_start[2]],
            color=connection_color,
            alpha=0.6,  # Increased opacity
            linewidth=abs(weight) * 4,  # Increased line width
        )
        ax.add_line(line)

        # Add arrow head
        arrow_mutation_scale = 20  # Size of arrow head
        ax.quiver(
            arrow_start[0],
            arrow_start[1],
            arrow_start[2],
            direction_normalized[0],
            direction_normalized[1],
            direction_normalized[2],
            length=arrow_length,
            color=connection_color,
            alpha=0.6,
            arrow_length_ratio=0.3,  # Controls the size of the arrow head
            normalize=True,
        )

    # Plot neurons
    scat = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c="gray",
        s=100,
    )

    # Highlight input and output neurons
    input_neurons = brain.input_neurons.cpu().numpy()
    output_neurons = brain.output_neurons.cpu().numpy()

    ax.scatter(
        positions[input_neurons, 0],
        positions[input_neurons, 1],
        positions[input_neurons, 2],
        c="yellow",
        s=150,
        label=f"Input neurons ({len(input_neurons)})",
    )

    ax.scatter(
        positions[output_neurons, 0],
        positions[output_neurons, 1],
        positions[output_neurons, 2],
        c="purple",
        s=150,
        label=f"Output neurons ({len(output_neurons)})",
    )

    # Add network stats to legend
    ax.scatter([], [], c="gray", s=100, label=f"Total neurons: {total_neurons}")
    ax.scatter([], [], c="gray", alpha=0, label=f"Total synapses: {total_synapses}")

    ax.legend()

    # Add a text annotation for the time step
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def update(frame):
        activations = activations_over_time[frame]
        colors = neuron_cmap(norm(activations))
        sizes = 100 + 100 * np.abs(activations)
        scat.set_color(colors)
        scat.set_sizes(sizes)
        time_text.set_text(f"Time Step: {frame + 1}/{time_steps}")
        return scat, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=time_steps, interval=100, blit=True, repeat=True
    )

    # Adjust the view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=45)
    ax.mouse_init()

    plt.title(
        "Neural Network Activity\nNeuron color = activation, Connection color = weight"
    )
    plt.tight_layout()
    plt.show()


def visualizeGradientFlowAsImage(brain, x, y):
    # Forward pass
    output = brain(x)
    loss = torch.nn.MSELoss()(output, y)

    dot = make_dot(loss, params=dict(brain.named_parameters()))

    # Save and display
    dot.render("computation_graph", format="png", cleanup=True)

    # Print additional info about gradients
    print("\nGradient flow info:")
    for name, param in brain.named_parameters():
        if param.grad is not None:
            print(f"{name}:")
            print(f"  Shape: {param.grad.shape}")
            print(f"  Gradient values: {param.grad}")
            print(f"  Requires grad: {param.requires_grad}")


def visualizeGradientFlowInBrowser(brain, x, y):
    writer = SummaryWriter("runs/experiment_1")

    # Add graph
    writer.add_graph(brain, x)

    print("\nTensorboard is ready!")
    print("Run 'tensorboard --logdir=runs' in terminal")
    print("Then open http://localhost:6006 in your browser")

    # Optional: add some extra parameter tracking
    for name, param in brain.named_parameters():
        if param.grad is not None:
            writer.add_histogram(name, param.data)
            writer.add_histogram(f"{name}.grad", param.grad)

    writer.close()
