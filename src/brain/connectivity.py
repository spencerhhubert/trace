def checkBidirectionalConnections(brain):
    connections = set()
    bidirectional = []
    for i in range(brain.synapse_indices.size(1)):
        from_idx = brain.synapse_indices[0, i].item()
        to_idx = brain.synapse_indices[1, i].item()
        if (to_idx, from_idx) in connections:
            bidirectional.append((from_idx, to_idx))
        connections.add((from_idx, to_idx))
    if bidirectional:
        print(f"Found {len(bidirectional)} bidirectional connections:")
        for a, b in bidirectional[:5]:  # Show first 5
            print(f"Between neurons {a} and {b}")
    return bidirectional

def analyzeConnectivity(brain):
    def findMinPathsToOutput(start_neuron):
        if start_neuron in brain.output_indices:
            return 0, [[start_neuron]]

        min_hops = float("inf")
        min_paths = []
        visited = set()
        queue = [(start_neuron, 0, [start_neuron])]  # (neuron, hops, path)
        visited.add(start_neuron)

        while queue:
            current, hops, path = queue.pop(0)

            if hops > min_hops:
                continue

            connections = brain.synapse_indices[1][
                brain.synapse_indices[0] == current
            ]

            for next_neuron in connections:
                next_neuron = next_neuron.item()
                if next_neuron in brain.output_indices:
                    new_path = path + [next_neuron]
                    if hops + 1 < min_hops:
                        min_hops = hops + 1
                        min_paths = [new_path]
                    elif hops + 1 == min_hops:
                        min_paths.append(new_path)
                elif next_neuron not in path:  # Allow revisiting nodes in different paths
                    if hops + 1 <= min_hops:  # Only explore if we haven't exceeded min_hops
                        queue.append((next_neuron, hops + 1, path + [next_neuron]))

        return min_hops, min_paths

    hop_counts = []
    for input_neuron in brain.input_indices:
        hops, paths = findMinPathsToOutput(input_neuron.item())
        if hops != float("inf"):
            hop_counts.append(hops)
            print(f"Input neuron {input_neuron}: {hops} hops to output, {len(paths)} distinct paths")
            for i, path in enumerate(paths):
                path_str = " -> ".join(
                    [
                        f"{n} (input)"
                        if n in brain.input_indices
                        else f"{n} (output)"
                        if n in brain.output_indices
                        else str(n)
                        for n in path
                    ]
                )
                print(f"  Path {i + 1}: {path_str}")
        else:
            print(f"Input neuron {input_neuron}: No path to output!")

    if hop_counts:
        print(f"\nAverage minimum hops: {sum(hop_counts) / len(hop_counts):.2f}")
        print(f"Min hops: {min(hop_counts)}")
        # print(f"Max hops: {max(hop_counts)}")
    else:
        print("\nNo valid paths from input to output!")
