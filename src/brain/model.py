import torch
from torch import nn
import os

NEURON_COUNT = 20
SYNAPSE_RATIO = 100
HIDDEN_NEURONS_LAYOUT = [5, 5]


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
        MAX_DISTANCE = 2
        # Create three clusters at different positions
        input_origin = torch.tensor([-1.0, -1.0, -1.0])
        hidden_origin = torch.tensor([0.0, 0.0, 0.0])
        output_origin = torch.tensor([1.0, 1.0, 1.0])

        # How spread out each cluster is
        HIDDEN_CLUSTER_SPREAD = 0.6
        IO_CLUSTER_SPREAD = 0.1

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

        # Calculate distances and connection probabilities
        distances = torch.cdist(self.neuron_positions, self.neuron_positions)
        probs = 1 / (
            distances**2 + 0.1
        )  # Reduce the denominator offset to make distant connections more likely
        probs[
            torch.arange(NEURON_COUNT), torch.arange(NEURON_COUNT)
        ] = 0  # No self connections

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

        print(f"Total connections: {len(rows)}")

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

        # Initialize neuron values and positions (even though positions aren't used in MLP)
        self.neuron_values = torch.zeros(NEURON_COUNT)
        self.neuron_positions = torch.zeros(NEURON_COUNT, 3)

    def step(self):
        next_values = torch.zeros_like(self.neuron_values)
        for i in range(len(self.synapse_weights)):
            from_idx = self.synapse_indices[0][i]
            to_idx = self.synapse_indices[1][i]
            next_values[to_idx] += (
                self.neuron_values[from_idx] * self.synapse_weights[i]
            )

        non_input_mask = torch.ones(NEURON_COUNT, dtype=bool, device=self.device)
        non_input_mask[self.input_indices] = False
        next_values[non_input_mask] += self.neuron_biases

        self.neuron_values = next_values

        mask = torch.ones(NEURON_COUNT, dtype=bool, device=self.device)
        mask[self.output_indices] = False
        self.neuron_values[mask] = torch.tanh(self.neuron_values[mask])

        # Store state after each step
        self.activation_history.append(self.neuron_values.clone().detach().cpu())

    def forward(self, x):
        self.activation_history = []  # Reset history for new forward pass
        self.neuron_values = torch.zeros(
            NEURON_COUNT, device=self.device, requires_grad=True
        )
        self.neuron_values.data[self.input_indices] = x.flatten()

        self.activation_history.append(self.neuron_values.clone().detach().cpu())

        STEPS = 3
        for _ in range(STEPS):
            self.step()

        return self.neuron_values[self.output_indices]

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

    def checkBidirectionalConnections(self):
        connections = set()
        bidirectional = []
        for i in range(self.synapse_indices.size(1)):
            from_idx = self.synapse_indices[0, i].item()
            to_idx = self.synapse_indices[1, i].item()
            if (to_idx, from_idx) in connections:
                bidirectional.append((from_idx, to_idx))
            connections.add((from_idx, to_idx))
        if bidirectional:
            print(f"Found {len(bidirectional)} bidirectional connections:")
            for a, b in bidirectional[:5]:  # Show first 5
                print(f"Between neurons {a} and {b}")
        return bidirectional

    def analyzeConnectivity(self):
        def minHopsToOutput(start_neuron):
            if start_neuron in self.output_indices:
                return 0, [start_neuron]

            visited = set()
            queue = [(start_neuron, 0, [start_neuron])]  # (neuron, hops, path)
            visited.add(start_neuron)

            while queue:
                current, hops, path = queue.pop(0)

                # Find all neurons this one connects to
                connections = self.synapse_indices[1][
                    self.synapse_indices[0] == current
                ]

                for next_neuron in connections:
                    next_neuron = next_neuron.item()
                    if next_neuron in self.output_indices:
                        return hops + 1, path + [next_neuron]
                    if next_neuron not in visited:
                        visited.add(next_neuron)
                        queue.append((next_neuron, hops + 1, path + [next_neuron]))

            return float("inf"), []  # No path found

        # Calculate min hops for each input neuron
        hop_counts = []
        for input_neuron in self.input_indices:
            hops, path = minHopsToOutput(input_neuron.item())
            if hops != float("inf"):
                hop_counts.append(hops)
                path_str = " -> ".join(
                    [
                        f"{n} (input)"
                        if n in self.input_indices
                        else f"{n} (output)"
                        if n in self.output_indices
                        else str(n)
                        for n in path
                    ]
                )
                print(f"Input neuron {input_neuron}: {hops} hops to output")
                print(f"Route: {path_str}")
            else:
                print(f"Input neuron {input_neuron}: No path to output!")

        if hop_counts:
            print(f"\nAverage minimum hops: {sum(hop_counts) / len(hop_counts):.2f}")
            print(f"Min hops: {min(hop_counts)}")
            print(f"Max hops: {max(hop_counts)}")
        else:
            print("\nNo valid paths from input to output!")
