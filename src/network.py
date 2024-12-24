import torch
from torch import nn
import numpy as np

INITIAL_NEURON_COUNT = 100
SYNAPSE_RATIO = 1000
MAX_DELAY_MS = 100
MAX_HISTORY_MS = 1000

class Brain(nn.Module):
    def __init__(self, device, neuron_count=INITIAL_NEURON_COUNT):
        super().__init__()
        self.neuron_count = neuron_count
        self.device = device

        # Learnable parameters
        self.connection_weights = nn.Parameter(torch.empty(0))

        # Non-learnable state
        self.positions = None
        self.connection_indices = None  # (2, num_connections) tensor
        self.delay_values = None        # (num_connections,) tensor
        self.activations = None
        self.activation_history = []
        self.time_step = 0

        self._initializeNetwork()
        self.to(device)

    def _initializeNetwork(self):
        cube_size = (self.neuron_count ** (1/3)) * 2
        self.positions = torch.rand((self.neuron_count, 3), device=self.device) * cube_size - (cube_size/2)

        # Calculate all pairwise distances
        diffs = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)
        distances = torch.norm(diffs, dim=2)

        # Create connectivity using power law
        connection_probs = 1 / (distances ** 2 + 1)
        connection_probs.fill_diagonal_(0)

        target_connections = self.neuron_count * SYNAPSE_RATIO
        connection_probs *= (target_connections / connection_probs.sum())

        # Sample connections based on probabilities
        random_mask = torch.rand_like(connection_probs) < connection_probs
        self.connection_indices = torch.nonzero(random_mask).t()  # (2, num_connections)

        # Initialize learnable weights
        self.connection_weights = nn.Parameter(
            torch.randn(self.connection_indices.shape[1], device=self.device) * 0.01
        )

        # Calculate delays based on distances
        from_idx = self.connection_indices[0]
        to_idx = self.connection_indices[1]
        self.delay_values = distances[from_idx, to_idx]

        self.activations = torch.zeros(self.neuron_count, device=self.device)

    def forward(self, input_data, steps=1):
        batch_size = input_data.shape[0]
        input_size = input_data.shape[1]

        # Handle each item in batch
        outputs = []
        for batch_idx in range(batch_size):
            # Reset state for each sequence
            self.activations = torch.zeros(self.neuron_count, device=self.device)
            self.activation_history = []

            # Initialize input neurons with this batch item
            self.activations[:input_size] = input_data[batch_idx]

            # Run network for specified steps
            for _ in range(steps):
                self.forwardStep()

            outputs.append(self.activations[-input_size:])

        return torch.stack(outputs)

    def forwardStep(self):
        self.time_step += 1
        self.activation_history.append(self.activations.clone())

        if len(self.activation_history) > MAX_HISTORY_MS:
            self.activation_history.pop(0)

        total_input = torch.zeros_like(self.activations)

        # Process all connections at once
        from_idx = self.connection_indices[0]
        to_idx = self.connection_indices[1]
        delays = self.delay_values.int()  # convert to integer delays

        for i, (from_i, to_i, delay, weight) in enumerate(zip(
            from_idx, to_idx, delays, self.connection_weights
        )):
            if delay < len(self.activation_history):
                total_input[to_i] += weight * self.activation_history[-delay][from_i]

        self.activations = torch.tanh(total_input)

    def addNeurons(self, count):
        old_count = self.neuron_count
        self.neuron_count += count

        # Extend positions
        cube_size = (self.neuron_count ** (1/3)) * 2
        new_positions = torch.rand((count, 3), device=self.device) * cube_size - (cube_size/2)
        self.positions = torch.cat([self.positions, new_positions])

        # Calculate new connections
        diffs = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)
        distances = torch.norm(diffs, dim=2)

        connection_probs = 1 / (distances ** 2 + 1)
        connection_probs.fill_diagonal_(0)
        connection_probs[:old_count, :old_count] = 0

        target_connections = count * SYNAPSE_RATIO
        connection_probs *= (target_connections / connection_probs.sum())

        random_mask = torch.rand_like(connection_probs) < connection_probs
        new_indices = torch.nonzero(random_mask).t()

        # Add new weights
        old_weights = self.connection_weights.data
        new_weights = torch.randn(new_indices.shape[1], device=self.device) * 0.01
        self.connection_weights = nn.Parameter(torch.cat([old_weights, new_weights]))

        # Update indices and delays
        self.connection_indices = torch.cat([self.connection_indices, new_indices], dim=1)
        from_idx = new_indices[0]
        to_idx = new_indices[1]
        new_delays = distances[from_idx, to_idx]
        self.delay_values = torch.cat([self.delay_values, new_delays])

        # Extend activations
        self.activations = torch.cat([
            self.activations,
            torch.zeros(count, device=self.device)
        ])

    def debugState(self):
        print(f"\nNetwork State:")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters())}")
        print(f"Connection weights range: {self.connection_weights.min():.3f} to {self.connection_weights.max():.3f}")
        print(f"Number of connections: {len(self.connection_weights)}")
        print(f"Activation history length: {len(self.activation_history)}")
        if self.activation_history:
            print(f"Recent activation range: {self.activation_history[-1].min():.3f} to {self.activation_history[-1].max():.3f}")
