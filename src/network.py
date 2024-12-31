import torch
from torch import nn

NEURON_COUNT = 12
SYNAPSE_RATIO = 100
MAX_NUM_TIMESTEPS_IN_HISTORY = 1
STEPS =  10
MAX_DISTANCE = 4  # None = unlimited, or set a number to limit connection distance
FORCE_MLP_STRUCTURE = True
HIDDEN_LAYERS = [5,5]  # Only used if FORCE_MLP_STRUCTURE is True


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
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.max_distance = MAX_DISTANCE
        self.history_size = MAX_NUM_TIMESTEPS_IN_HISTORY
        self.activations_over_time = []
        self.store_activations = True

        if FORCE_MLP_STRUCTURE:
            self.neuron_count = input_size + sum(HIDDEN_LAYERS) + output_size
            if self.neuron_count != neuron_count:
                raise ValueError(
                    f"NEURON_COUNT ({neuron_count}) must equal sum of layers "
                    f"({input_size} + {sum(HIDDEN_LAYERS)} + {output_size} = {self.neuron_count})"
                )
        else:
            self.neuron_count = neuron_count

        self.act = torch.nn.Tanh()

        if FORCE_MLP_STRUCTURE:
            self._initMLPPositions()
            self._initMLPConnections()
        else:
            self._initPositions(cluster_io)
            self._initConnections()
            self._pruneIsolatedNeurons()

        self._initWeights()
        self._initStateVariables()
        if not FORCE_MLP_STRUCTURE:
            self._severInputOutputConnections()
        self.to(device)

    def _initMLPPositions(self):
        # Calculate the total number of layers (input + hidden + output)
        total_layers = len(HIDDEN_LAYERS) + 2
        layer_sizes = [self.input_size] + HIDDEN_LAYERS + [self.output_size]

        # Initialize positions tensor
        self.positions = torch.zeros((self.neuron_count, 3), device=self.device)

        # Set z-coordinate based on layer (depth)
        current_idx = 0
        z_spacing = 2.0  # Space between layers
        for layer_idx, size in enumerate(layer_sizes):
            # Z coordinate is uniform for each layer
            z = layer_idx * z_spacing - (total_layers - 1) * z_spacing / 2

            # Arrange neurons in a grid pattern on the XY plane
            grid_size = int(torch.sqrt(torch.tensor(size)).ceil())
            for i in range(size):
                x = (i % grid_size) - grid_size / 2
                y = (i // grid_size) - grid_size / 2
                self.positions[current_idx + i] = torch.tensor([x, y, z])

            current_idx += size

    def _initMLPConnections(self):
        connections = []
        layer_sizes = [self.input_size] + HIDDEN_LAYERS + [self.output_size]

        # Keep track of the starting index for each layer
        current_idx = 0
        next_idx = self.input_size

        # Connect each layer to the next layer
        for i in range(len(layer_sizes) - 1):
            current_size = layer_sizes[i]
            next_size = layer_sizes[i + 1]

            # Fully connect current layer to next layer
            for j in range(current_size):
                for k in range(next_size):
                    connections.append([
                        current_idx + j,  # From neuron in current layer
                        next_idx + k      # To neuron in next layer
                    ])

            current_idx = next_idx
            next_idx += next_size

        self.connection_indices = torch.tensor(connections, device=self.device).t()

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
            * 0.01
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

    # note: because of this behavior where we stop the forward pass when an output neuron is triggered
    # first of all, we're going to need to something clever. this is kind of a big architectural mess
    # but also, I think we can simple force a min number of jumps between the input and output neurons
    def step(self, should_print=False):
        if should_print:
            active_indices = torch.nonzero(self.activations).squeeze()
            if active_indices.dim() == 0:
                active_indices = active_indices.unsqueeze(0)
            active_values = self.activations[active_indices]
            print(f"\n=== Step {self.time_step + 1} ===")
            print("Active neurons before step:")
            for idx, val in zip(active_indices.cpu().tolist(), active_values.cpu().tolist()):
                print(f"  Neuron {idx}: {val:.3f}")

        # Compute next timestep (starts as all zeros)
        next_activations = torch.zeros_like(self.activations)

        # Get all connections and their current values
        from_idx = self.connection_indices[0]
        to_idx = self.connection_indices[1]
        from_values = self.activations[from_idx]

        if should_print:
            print("\nActive connections:")
            active_conns = from_values != 0
            for i, is_active in enumerate(active_conns):
                if is_active:
                    print(f"  {from_idx[i]} -> {to_idx[i]}: weight={self.connection_weights[i]:.3f}, value={from_values[i]:.3f}")

        # Compute weighted inputs to each target neuron
        weighted_inputs = from_values * self.connection_weights
        next_activations.index_add_(0, to_idx, weighted_inputs)

        if should_print:
            print("\nRaw inputs to neurons (before bias/activation):")
            receiving_indices = torch.nonzero(next_activations != 0).squeeze()
            if receiving_indices.dim() == 0:
                receiving_indices = receiving_indices.unsqueeze(0)
            for idx in receiving_indices:
                print(f"  Neuron {idx}: {next_activations[idx]:.3f}")

        # Only add bias and apply activation to neurons that received input
        active_neurons = next_activations != 0
        next_activations[active_neurons] += self.biases[active_neurons]

        # Don't apply activation to output neuron
        active_hidden_neurons = active_neurons.clone()
        active_hidden_neurons[-self.output_size:] = False

        if active_hidden_neurons.any():
            active_idx = torch.nonzero(active_hidden_neurons).squeeze()
            if active_idx.dim() == 0:
                active_idx = active_idx.unsqueeze(0)
            pre_act = next_activations[active_hidden_neurons]
            next_activations[active_hidden_neurons] = self.act(next_activations[active_hidden_neurons])
            post_act = next_activations[active_hidden_neurons]

            if should_print:
                print("\nAfter bias and activation:")
                for idx, pre, post in zip(active_idx, pre_act, post_act):
                    print(f"  Neuron {idx}: {pre:.3f} -> {post:.3f}")

        # Previous activations are completely replaced by new ones
        self.activation_history.append(next_activations.clone())
        self.activations = next_activations
        self.time_step += 1

        output_received_signal = next_activations[-1] != 0
        if output_received_signal and should_print:
            print(f"\nOutput neuron activated: {next_activations[-1]:.3f}")

        return output_received_signal

    def forward(self, input_data):
        batch_size = input_data.shape[0]
        outputs = []

        for batch_idx in range(batch_size):
            # Reset network state
            self.activation_history = []
            self.activations = torch.zeros(self.neuron_count, device=self.device)
            self.time_step = 0

            # Set input value only at timestep 0
            self.activations[: self.input_size] = input_data[batch_idx]
            self.activation_history.append(self.activations.clone())

            # Step until output neuron activates or max steps reached
            for _ in range(STEPS):
                if self.step(should_print=False):
                    break

            outputs.append(self.activations[-1].clone())

        # Store full activation history for visualization
        self.activations_over_time = torch.stack(self.activation_history)

        return torch.stack(outputs)
