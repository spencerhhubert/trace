import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

NEURON_COUNT = 20
SYNAPSE_RATIO = 100
MAX_NUM_TIMESTEPS_IN_HISTORY = 1
STEPS = 20
MAX_DISTANCE = 5 # None = unlimited, or set a number to limit connection distance


class Brain(nn.Module):
    def __init__(
        self,
        device,
        neuron_count,
        input_size,
        output_size,
        cluster_io,
    ):
        super().__init__()
        self.neuron_count = neuron_count
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.max_distance = MAX_DISTANCE
        self.history_size = MAX_NUM_TIMESTEPS_IN_HISTORY
        self.activations_over_time = []

        self._initPositions(cluster_io)
        self._initConnections()
        self._pruneIsolatedNeurons()
        self._initWeights()
        self._initStateVariables()
        self._severInputOutputConnections()
        self.to(device)

    def _severInputOutputConnections(self):
        # Special case: remove any direct connections between input and output neurons
        mask = torch.ones(
            self.connection_indices.shape[1], dtype=torch.bool, device=self.device
        )

        for i, (from_idx, to_idx) in enumerate(
            zip(self.connection_indices[0], self.connection_indices[1])
        ):
            # Check if connection is from input to output
            is_from_input = from_idx < self.input_size
            is_to_output = to_idx >= (self.neuron_count - self.output_size)
            if is_from_input and is_to_output:
                mask[i] = False

        # Keep only connections that aren't direct input->output
        self.connection_indices = self.connection_indices[:, mask]
        self.connection_weights = nn.Parameter(self.connection_weights[mask])

    def _initPositions(self, cluster_io):
        cube_size = (self.neuron_count ** (1 / 3)) * 2
        self.positions = torch.rand(
            (self.neuron_count, 3), device=self.device
        ) * cube_size - (cube_size / 2)

        if cluster_io:
            # Cluster input neurons at one corner
            self.positions[: self.input_size] = torch.rand(
                (self.input_size, 3), device=self.device
            ) * (cube_size * 0.1) - (cube_size / 2)

            # Cluster output neurons at the opposite corner
            self.positions[-self.output_size :] = torch.rand(
                (self.output_size, 3), device=self.device
            ) * (cube_size * 0.1) + (cube_size / 2 - cube_size * 0.1)

    def _initConnections(self):
        # Calculate all pairwise distances
        diffs = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)
        distances = torch.norm(diffs, dim=2)

        # Create separate probability matrices for input, output, and hidden connections
        input_probs = torch.zeros_like(distances)
        input_probs[: self.input_size, self.input_size :] = 1 / (
            distances[: self.input_size, self.input_size :] ** 2 + 1
        )

        output_probs = torch.zeros_like(distances)
        output_probs[self.input_size : -self.output_size, -self.output_size :] = 1 / (
            distances[self.input_size : -self.output_size, -self.output_size :] ** 2 + 1
        )

        hidden_probs = torch.zeros_like(distances)
        hidden_probs[
            self.input_size : -self.output_size, self.input_size : -self.output_size
        ] = 1 / (
            distances[
                self.input_size : -self.output_size, self.input_size : -self.output_size
            ]
            ** 2
            + 1
        )

        if self.max_distance is not None:
            input_probs[distances > self.max_distance] = 0
            output_probs[distances > self.max_distance] = 0
            hidden_probs[distances > self.max_distance] = 0

        # Calculate target connections for each section
        target_input_connections = self.input_size * SYNAPSE_RATIO
        target_output_connections = self.output_size * SYNAPSE_RATIO
        target_hidden_connections = (
            self.neuron_count - self.input_size - self.output_size
        ) * SYNAPSE_RATIO

        # Normalize each section separately
        input_probs *= target_input_connections / (input_probs.sum() + 1e-10)
        output_probs *= target_output_connections / (output_probs.sum() + 1e-10)
        hidden_probs *= target_hidden_connections / (hidden_probs.sum() + 1e-10)

        # Combine probabilities
        connection_probs = input_probs + output_probs + hidden_probs

        # Track connected pairs to prevent bidirectional connections
        connected_pairs = set()

        # Sample connections while respecting the no-bidirectional rule
        connections = []
        for i in range(self.neuron_count):
            for j in range(self.neuron_count):
                if i == j:  # Skip self-connections
                    continue

                # Skip if either direction is already connected
                if (i, j) in connected_pairs or (j, i) in connected_pairs:
                    continue

                # Sample based on probability
                if torch.rand(1).item() < connection_probs[i, j].item():
                    connections.append([i, j])
                    connected_pairs.add((i, j))

        if not connections:
            raise ValueError("No valid connections were created!")

        self.connection_indices = torch.tensor(connections, device=self.device).t()

        # Store distances for delay calculations
        self.distances = distances

    def _pruneIsolatedNeurons(self):
        # Create adjacency lists for forward and backward traversal
        forward_adj = [[] for _ in range(self.neuron_count)]
        backward_adj = [[] for _ in range(self.neuron_count)]

        for from_idx, to_idx in zip(
            self.connection_indices[0], self.connection_indices[1]
        ):
            forward_adj[from_idx.item()].append(to_idx.item())
            backward_adj[to_idx.item()].append(from_idx.item())

        # Find all neurons reachable from inputs
        reachable_from_input = set()
        stack = list(range(self.input_size))
        while stack:
            node = stack.pop()
            if node not in reachable_from_input:
                reachable_from_input.add(node)
                stack.extend(forward_adj[node])

        # Find all neurons that can reach outputs
        can_reach_output = set()
        stack = list(range(self.neuron_count - self.output_size, self.neuron_count))
        while stack:
            node = stack.pop()
            if node not in can_reach_output:
                can_reach_output.add(node)
                stack.extend(backward_adj[node])

        # Keep only neurons that are both reachable from input and can reach output
        valid_neurons = reachable_from_input.intersection(can_reach_output)
        valid_neurons = valid_neurons.union(
            set(range(self.input_size))
        )  # Always keep input neurons
        valid_neurons = valid_neurons.union(
            set(range(self.neuron_count - self.output_size, self.neuron_count))
        )  # Always keep output neurons

        # Create mapping from old to new indices
        valid_neurons = sorted(list(valid_neurons))
        old_to_new = {old: new for new, old in enumerate(valid_neurons)}

        # Update connections
        valid_connections = []
        for from_idx, to_idx in zip(
            self.connection_indices[0], self.connection_indices[1]
        ):
            from_idx, to_idx = from_idx.item(), to_idx.item()
            if from_idx in valid_neurons and to_idx in valid_neurons:
                valid_connections.append([old_to_new[from_idx], old_to_new[to_idx]])

        # Update network properties
        if valid_connections:
            self.connection_indices = torch.tensor(
                valid_connections, device=self.device
            ).t()
        else:
            raise ValueError("No valid connections remain after pruning!")

        self.positions = self.positions[valid_neurons]
        self.neuron_count = len(valid_neurons)

    def _initWeights(self):
        # Initialize learnable weights
        self.connection_weights = nn.Parameter(
            torch.randn(self.connection_indices.shape[1], device=self.device)
            * 0.1  # 0.01
        )

    def _initStateVariables(self):
        self.activations = torch.zeros(self.neuron_count, device=self.device)
        self.activation_history = []
        self.biases = nn.Parameter(
            torch.zeros(self.neuron_count, device=self.device) * 0.1
        )
        self.time_step = 0

        # Input and output neuron indices
        self.input_neurons = torch.arange(self.input_size, device=self.device)
        self.output_neurons = torch.arange(
            self.neuron_count - self.output_size, self.neuron_count, device=self.device
        )

    def forward(self, input_data):
        batch_size = input_data.shape[0]
        self.store_activations = True  # Flag to control activation recording

        outputs = []
        for batch_idx in range(batch_size):
            # Reset state for each sequence
            self.activations = torch.zeros(self.neuron_count, device=self.device)
            self.activation_history = []
            self.time_step = 0

            # Store current input for use in step
            self.current_input = input_data[batch_idx]
            # Initialize input neurons with this batch item
            self.activations[: self.input_size] = self.current_input

            for _ in range(STEPS):
                self.step()

            outputs.append(self.activations[-self.output_size:].clone())

        self.activations_over_time = (
            torch.stack(self.activation_history)
            if self.activation_history
            else None
        )

        return torch.stack(outputs)

    def step(self):
        self.time_step += 1
        prev_activations = self.activations

        # Optionally store activations for analysis without tracking gradients
        if self.store_activations:
            self.activation_history.append(prev_activations.detach().cpu())

        from_idx = self.connection_indices[0]
        to_idx = self.connection_indices[1]

        activations_from = prev_activations[from_idx]
        weights = self.connection_weights

        weighted_inputs = activations_from * weights

        # Use out-of-place operation to prevent interfering with autograd
        total_input = torch.zeros_like(self.activations)
        total_input = total_input.index_add(0, to_idx, weighted_inputs)

        total_input += self.biases

        # Compute new activations
        new_activations = torch.tanh(total_input)

        # Accumulate activations over time
        self.activations = prev_activations + new_activations

        # Optionally apply decay to prevent activations from growing indefinitely
        # Uncomment the next line to apply decay
        # self.activations = self.activations * decay_factor

        # Keep input neurons' activations fixed to the input data
        input_indices = torch.arange(self.input_size, device=self.device)
        self.activations[input_indices] = self.current_input
