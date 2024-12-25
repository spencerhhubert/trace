import torch
from torch import nn

INITIAL_NEURON_COUNT = 100
SYNAPSE_RATIO = 1000
MAX_DELAY_MS = 100
MAX_HISTORY_MS = 1000

USE_DELAYS = True
MAX_DISTANCE = None  # None = unlimited, or set a number to limit connection distance
HISTORY_SIZE = 1000  # Could reduce to 1 to test without delays

class Brain(nn.Module):
    def __init__(self, device, neuron_count=INITIAL_NEURON_COUNT, use_delays=True, max_distance=None, history_size=MAX_HISTORY_MS):
        super().__init__()
        self.neuron_count = neuron_count
        self.device = device
        self.use_delays = use_delays
        self.max_distance = max_distance
        self.history_size = history_size

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

        # Apply distance limit if specified
        if self.max_distance is not None:
            connection_probs = 1 / (distances ** 2 + 1)
            connection_probs[distances > self.max_distance] = 0
        else:
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
        if self.use_delays:
            self.delay_values = distances[from_idx, to_idx]
        else:
            self.delay_values = torch.ones_like(distances[from_idx, to_idx])

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

        if len(self.activation_history) > self.history_size:
            self.activation_history.pop(0)

        total_input = torch.zeros_like(self.activations)

        # Process all connections at once
        from_idx = self.connection_indices[0]
        to_idx = self.connection_indices[1]

        if self.use_delays:
            delays = self.delay_values.int()
            for i, (from_i, to_i, delay, weight) in enumerate(zip(
                from_idx, to_idx, delays, self.connection_weights
            )):
                if delay < len(self.activation_history):
                    total_input[to_i] += weight * self.activation_history[-delay][from_i]
        else:
            # Direct connections without delay
            total_input.index_add_(0, to_idx, self.connection_weights * self.activations[from_idx])

        self.activations = torch.tanh(total_input)
