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
        # Generate random positions for all neurons in 3D space
        SPREAD_FACTOR = 5
        self.neuron_positions = (torch.rand(NEURON_COUNT, 3) * 2 - 1) * SPREAD_FACTOR

        # Select input and output neurons from opposite ends of x-axis
        sorted_x = torch.argsort(self.neuron_positions[:, 0])
        self.input_indices = sorted_x[: self.input_size]
        self.output_indices = sorted_x[-self.output_size :]

        # Calculate distances between all neurons
        distances = torch.cdist(self.neuron_positions, self.neuron_positions)

        # Generate connection probabilities based on inverse square distance
        probs = 1 / (distances**2 + 1)
        probs[
            torch.arange(NEURON_COUNT), torch.arange(NEURON_COUNT)
        ] = 0  # No self connections

        # Calculate actual possible connections vs target
        target_connections = int(NEURON_COUNT * SYNAPSE_RATIO)
        possible_connections = NEURON_COUNT * (NEURON_COUNT - 1)
        actual_connections = min(target_connections, possible_connections)

        flat_probs = probs.flatten()
        sampled_indices = torch.multinomial(flat_probs, actual_connections)

        rows = sampled_indices // NEURON_COUNT
        cols = sampled_indices % NEURON_COUNT

        self.synapse_indices = torch.stack([rows, cols])
        self.synapse_weights = nn.Parameter(torch.randn(actual_connections))

        # Initialize biases (skip input neurons)
        non_input_mask = torch.ones(NEURON_COUNT, dtype=bool)
        non_input_mask[self.input_indices] = False
        self.neuron_biases = nn.Parameter(torch.randn(NEURON_COUNT)[non_input_mask])

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

    def forward(self, x):
        self.activation_history = []  # Reset history for new forward pass
        self.neuron_values = torch.zeros(
            NEURON_COUNT, device=self.device, requires_grad=True
        )
        self.neuron_values.data[self.input_indices] = x.flatten()

        self.activation_history.append(self.neuron_values.clone().detach().cpu())

        for step in range(3):
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

        return self.neuron_values[self.output_indices]

    def save(self, filepath):
        state = {
            "neuron_positions": self.neuron_positions,
            "input_indices": self.input_indices,
            "output_indices": self.output_indices,
            "synapse_indices": self.synapse_indices,
            "state_dict": self.state_dict(),  # This saves weights and biases
            "init_strategy": self.init_strategy,
            "input_size": self.input_size,
            "output_size": self.output_size,
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
