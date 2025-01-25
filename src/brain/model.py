import torch
from torch import nn
import os

NEURON_COUNT = 7
SYNAPSE_RATIO = 100
HIDDEN_NEURONS_LAYOUT = [5]


class Brain(nn.Module):
    def __init__(self, device, input_size, output_size, init_strategy):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.init_strategy = init_strategy

        self.neuron_positions = None
        self.neuron_values = None
        self.input_indices = None
        self.output_indices = None
        self.synapse_indices = None
        self.synapse_weights = None
        self.activation_history = []

        if init_strategy == "spatial":
            self._initSpatial()
        elif init_strategy == "mlp":
            self._initMLP()
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

        self.to(device)

    def _initSpatial(self):
        MAX_DISTANCE = 4
        SCALE = 2.5
        # Create three clusters at different positions
        input_origin = torch.tensor([-1.0, -1.0, -1.0]) * SCALE
        hidden_origin = torch.tensor([0.0, 0.0, 0.0])
        output_origin = torch.tensor([1.0, 1.0, 1.0]) * SCALE

        # How spread out each cluster is
        HIDDEN_CLUSTER_SPREAD = 0.8
        IO_CLUSTER_SPREAD = 0.0

        # how much to stretch the hidden cluster along the line between input and output
        STRETCH_FACTOR = 2.0  # 1.0 means no stretch

        # Place neurons in their clusters
        self.neuron_positions = torch.empty(NEURON_COUNT, 3)
        self.input_indices = torch.arange(self.input_size)
        self.output_indices = torch.arange(
            NEURON_COUNT - self.output_size, NEURON_COUNT
        )

        # Create each cluster
        self.neuron_positions[self.input_indices] = (
            torch.randn(self.input_size, 3) * IO_CLUSTER_SPREAD + input_origin
        )
        self.neuron_positions[self.output_indices] = (
            torch.randn(self.output_size, 3) * IO_CLUSTER_SPREAD + output_origin
        )

        # Hidden neurons
        hidden_start = self.input_size
        hidden_end = NEURON_COUNT - self.output_size
        n_hidden = hidden_end - hidden_start
        self.neuron_positions[hidden_start:hidden_end] = (
            torch.randn(n_hidden, 3) * HIDDEN_CLUSTER_SPREAD + hidden_origin
        )

        # Stretch hidden neurons along i/o axis
        direction = output_origin - input_origin
        direction = direction / torch.norm(direction)
        hidden_positions = self.neuron_positions[hidden_start:hidden_end]
        proj = torch.matmul(hidden_positions, direction.unsqueeze(0).T)
        self.neuron_positions[hidden_start:hidden_end] = (
            hidden_positions + direction.unsqueeze(0) * proj * (STRETCH_FACTOR - 1.0)
        )

        # Calculate distances and connection probabilities
        distances = torch.cdist(self.neuron_positions, self.neuron_positions)

        # Calculate directional bias. we increase the chance that the direction flows from the input to the output
        direction = output_origin - input_origin
        direction = direction / torch.norm(direction)

        # Calculate connection directions
        connection_directions = self.neuron_positions.unsqueeze(0) - self.neuron_positions.unsqueeze(1)
        connection_directions = connection_directions / (torch.norm(connection_directions, dim=2, keepdim=True) + 1e-6)

        # Dot product with desired direction shows alignment (-1 to 1)
        alignment = torch.sum(connection_directions * direction, dim=2)
        alignment = (alignment + 1) / 2  # convert to 0 to 1

        DIRECTION_BIAS = 3.0  # How much to favor aligned connections
        probs = (1 / (distances**2 + 0.1)) * (1 + alignment * DIRECTION_BIAS)
        probs[torch.arange(NEURON_COUNT), torch.arange(NEURON_COUNT)] = 0  # No self connections

        # Create all possible connections (only one direction between any two neurons)
        rows, cols = [], []
        for i in range(NEURON_COUNT):
            for j in range(i + 1, NEURON_COUNT):  # Only upper triangle
                if (
                    distances[i, j] <= MAX_DISTANCE
                ):  # Only consider connections within MAX_DISTANCE
                    rows.append(i)
                    cols.append(j)

        if not rows:  # If no valid connections possible
            raise ValueError(
                f"No valid connections possible with MAX_DISTANCE={MAX_DISTANCE}. Try increasing MAX_DISTANCE or adjusting cluster positions."
            )

        # Calculate probability for each possible connection
        indices = torch.tensor([rows, cols])
        connection_distances = distances[indices[0], indices[1]]
        connection_probs = 1 / (connection_distances**2 + 0.1)

        # Sample connections
        target_connections = min(int(NEURON_COUNT * SYNAPSE_RATIO), len(rows))
        sampled_idx = torch.multinomial(connection_probs, target_connections)

        rows = torch.tensor(rows)[sampled_idx]
        cols = torch.tensor(cols)[sampled_idx]

        # flip i/o connections such that input only fan out and output only fan in
        for i in range(len(rows)):
            from_idx, to_idx = rows[i], cols[i]
            if (
                from_idx in self.output_indices or to_idx in self.input_indices
            ) and from_idx != to_idx:
                rows[i], cols[i] = cols[i], rows[i]

        # Remove any self connections that might have been created
        valid_connections = rows != cols
        rows = rows[valid_connections]
        cols = cols[valid_connections]

        self.synapse_indices = torch.stack([rows, cols])
        self.synapse_weights = nn.Parameter(torch.randn(len(rows)))

        # Initialize biases (skip input neurons)
        non_input_mask = torch.ones(NEURON_COUNT, dtype=bool)
        non_input_mask[self.input_indices] = False
        self.neuron_biases = nn.Parameter(torch.randn(NEURON_COUNT)[non_input_mask])

        self.trimToMinimalPaths()

        print(f"Total connections: {len(rows)}")

    def trimToMinimalPaths(self):
        # First get all minimal paths
        needed_neurons = set()
        needed_connections = set()

        for input_n in self.input_indices:
            min_hops = float('inf')
            min_paths = []
            queue = [(input_n.item(), 0, [input_n.item()])]  # (node, hops, path)

            while queue:
                current, hops, path = queue.pop(0)
                if hops > min_hops:
                    continue

                if current in self.output_indices:
                    if hops < min_hops:
                        min_hops = hops
                        min_paths = [path]
                    elif hops == min_hops:
                        min_paths.append(path)
                    continue

                next_neurons = self.synapse_indices[1][self.synapse_indices[0] == current]
                for next_n in next_neurons:
                    next_n = next_n.item()
                    if next_n not in path:
                        queue.append((next_n, hops + 1, path + [next_n]))

            # Add all neurons and connections from minimal paths
            for path in min_paths:
                for n in path:
                    needed_neurons.add(n)
                for i in range(len(path)-1):
                    needed_connections.add((path[i], path[i+1]))

        if not needed_neurons:
            print("No valid paths found!")
            return

        # Create new mapping for needed neurons
        old_to_new = {old: new for new, old in enumerate(sorted(needed_neurons))}

        # Filter connections and weights
        new_connections = []
        new_weights = []
        for i in range(len(self.synapse_indices[0])):
            from_n = self.synapse_indices[0][i].item()
            to_n = self.synapse_indices[1][i].item()
            if (from_n, to_n) in needed_connections:
                new_connections.append([old_to_new[from_n], old_to_new[to_n]])
                new_weights.append(self.synapse_weights[i])

        # Update model
        self.synapse_indices = torch.tensor(new_connections).t()
        self.synapse_weights = nn.Parameter(torch.stack(new_weights))
        self.neuron_positions = torch.stack([self.neuron_positions[i] for i in sorted(needed_neurons)])
        self.input_indices = torch.tensor([old_to_new[i.item()] for i in self.input_indices])
        self.output_indices = torch.tensor([old_to_new[i.item()] for i in self.output_indices])

        # Update biases
        non_input_mask = torch.ones(len(needed_neurons), dtype=bool)
        non_input_mask[self.input_indices] = False
        self.neuron_biases = nn.Parameter(torch.randn(non_input_mask.sum()))

        global NEURON_COUNT
        NEURON_COUNT = len(needed_neurons)
        print(f"Trimmed to {NEURON_COUNT} neurons")


    def _initMLP(self):
        expected_neurons = (
            self.input_size + self.output_size + sum(HIDDEN_NEURONS_LAYOUT)
        )
        if expected_neurons != NEURON_COUNT:
            raise ValueError(
                f"NEURON_COUNT ({NEURON_COUNT}) doesn't match expected neurons for MLP layout ({expected_neurons})"
            )

        # Assign neuron indices for each layer
        current_idx = 0
        self.input_indices = torch.arange(current_idx, current_idx + self.input_size)
        current_idx += self.input_size

        hidden_indices = []
        for layer_size in HIDDEN_NEURONS_LAYOUT:
            hidden_indices.append(torch.arange(current_idx, current_idx + layer_size))
            current_idx += layer_size

        self.output_indices = torch.arange(current_idx, current_idx + self.output_size)

        # Initialize positions tensor
        self.neuron_positions = torch.zeros((NEURON_COUNT, 3))

        # Create fully connected layers
        connections = []
        weights = []

        # Input to first hidden
        for i in self.input_indices:
            for j in hidden_indices[0]:
                connections.append([i, j])
                weights.append(torch.randn(1))

        # Hidden to hidden
        for h1, h2 in zip(hidden_indices[:-1], hidden_indices[1:]):
            for i in h1:
                for j in h2:
                    connections.append([i, j])
                    weights.append(torch.randn(1))

        # Last hidden to output
        for i in hidden_indices[-1]:
            for j in self.output_indices:
                connections.append([i, j])
                weights.append(torch.randn(1))

        self.synapse_indices = torch.tensor(connections).t()
        self.synapse_weights = nn.Parameter(torch.cat(weights))

        # Initialize biases (skip input neurons)
        non_input_mask = torch.ones(NEURON_COUNT, dtype=bool)
        non_input_mask[self.input_indices] = False
        self.neuron_biases = nn.Parameter(torch.randn(NEURON_COUNT)[non_input_mask])

        # Space out neuron positions to match MLP layout
        layer_counts = [self.input_size] + HIDDEN_NEURONS_LAYOUT + [self.output_size]
        z_spacing = 2.0 / (len(layer_counts) - 1)  # Total depth of 2 units

        current_idx = 0
        for layer_idx, layer_size in enumerate(layer_counts):
            # Calculate vertical spacing for this layer
            y_spacing = 2.0 / max(layer_size - 1, 1)  # Max height of 2 units
            for neuron_idx in range(layer_size):
                y_pos = -1.0 + neuron_idx * y_spacing if layer_size > 1 else 0
                self.neuron_positions[current_idx] = torch.tensor([
                    0.0,  # x
                    y_pos,  # y
                    -1.0 + layer_idx * z_spacing  # z
                ])
                current_idx += 1

        # Initialize neuron values
        self.neuron_values = torch.zeros(NEURON_COUNT)

    def step(self):
        next_values = torch.zeros_like(self.neuron_values)
        neurons_activated = torch.zeros((self.neuron_values.shape[1],), dtype=bool, device=self.device)

        for i in range(len(self.synapse_weights)):
            from_idx = self.synapse_indices[0][i]
            to_idx = self.synapse_indices[1][i]
            next_values[:, to_idx] += self.neuron_values[:, from_idx] * self.synapse_weights[i]
            neurons_activated[to_idx] = True

        non_input_mask = torch.ones(NEURON_COUNT, dtype=bool, device=self.device)
        non_input_mask[self.input_indices] = False
        next_values[:, non_input_mask] += self.neuron_biases

        # Only apply activation to neurons that received input (excluding output neurons)
        activation_mask = neurons_activated & ~torch.isin(torch.arange(NEURON_COUNT, device=self.device), self.output_indices)
        next_values[:, activation_mask] = torch.tanh(next_values[:, activation_mask])

        self.neuron_values = next_values
        # Store only the first sample's activations for visualization
        self.activation_history.append(self.neuron_values[0].clone().detach().cpu())

    def forward(self, x):
        self.activation_history = []
        batch_size = x.shape[0]

        # Create neuron values for the whole batch
        self.neuron_values = torch.zeros(
            (batch_size, NEURON_COUNT), device=self.device, requires_grad=True
        )

        # Create a new tensor instead of modifying in place
        values = self.neuron_values.clone()
        values[:, self.input_indices] = x
        self.neuron_values = values

        self.activation_history.append(self.neuron_values.clone().detach().cpu())

        STEPS = 3
        for _ in range(STEPS):
            self.step()

        return self.neuron_values[:, self.output_indices]

    def save(self, filepath):
        state = {
            "neuron_positions": self.neuron_positions,
            "input_indices": self.input_indices,
            "output_indices": self.output_indices,
            "synapse_indices": self.synapse_indices,
            "state_dict": self.state_dict(),
            "init_strategy": self.init_strategy,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "n_connections": self.synapse_weights.shape[0],
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(state, filepath)

    @classmethod
    def load(cls, filepath, device):
        state = torch.load(filepath)
        brain = cls(
            device=device,
            input_size=state["input_size"],
            output_size=state["output_size"],
            init_strategy=state["init_strategy"],
        )

        brain.neuron_positions = state["neuron_positions"]
        brain.input_indices = state["input_indices"]
        brain.output_indices = state["output_indices"]
        brain.synapse_indices = state["synapse_indices"]
        brain.load_state_dict(state["state_dict"])

        return brain
