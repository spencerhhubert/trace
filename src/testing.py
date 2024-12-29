from plotting import plotTrainingMetrics
from training import *
import torch
import matplotlib.pyplot as plt
import numpy as np


def trainSinX(brain, n_epochs):
    x = torch.linspace(-2 * np.pi, 2 * np.pi, 200).unsqueeze(1)
    y = torch.sin(x)

    x = (x + 2 * np.pi) / (4 * np.pi)
    y = (y + 1) / 2

    #overwrite to just be linear
    x = torch.linspace(0, 1, 200).unsqueeze(1)
    y = 2 * x + 0.3  # y = 2x + 0.3

    x = x.to(brain.device)
    y = y.to(brain.device)

    metrics = trainNetwork(brain, x, y, n_epochs)
    return metrics, x, y


def plotSinXResults(brain, x, y):
    plt.figure(figsize=(10, 5))
    plt.plot(x.cpu().numpy(), y.cpu().numpy(), label="True sin(x)")
    with torch.no_grad():
        y_pred = torch.cat([brain(x[i : i + 10]) for i in range(0, len(x), 10)])
        plt.plot(x.cpu().numpy(), y_pred.cpu().numpy(), "--", label="Network output")
    plt.legend()
    plt.grid(True)
    plt.show()


def trainIdentity(brain):
    x = torch.linspace(0, 1, 100).unsqueeze(1)
    y = x.clone()

    x = x.to(brain.device)
    y = y.to(brain.device)

    print("\nTraining on identity function (y=x)...")
    metrics = trainNetwork(brain, x, y, n_epochs=100)
    return metrics


def trainMNIST(brain, train_loader, n_epochs=10):
    optimizer = torch.optim.Adam(brain.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(
        brain.device
    )  # Move criterion to same device as brain

    metrics = {
        "loss": [],
        "grad_norm": [],
        "activation_mean": [],
        "activation_std": [],
        "active_neuron_ratio": [],
        "accuracy": [],
    }

    pbar = tqdm(range(n_epochs), desc="Training MNIST")
    for epoch in pbar:
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(brain.device), target.to(brain.device)
            optimizer.zero_grad()

            batch_size = data.size(0)
            data = data.view(batch_size, -1)  # flatten images

            output = brain(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            total_loss += loss.item()

            # Collect debug metrics
            with torch.no_grad():
                grad_norm = torch.norm(
                    torch.stack(
                        [
                            p.grad.norm()
                            for p in brain.parameters()
                            if p.grad is not None
                        ]
                    )
                )

                active_neurons = (brain.activations.abs() > 0.01).float().mean()
                act_mean = brain.activations.mean()
                act_std = brain.activations.std()

                metrics["grad_norm"].append(grad_norm.item())
                metrics["activation_mean"].append(act_mean.item())
                metrics["activation_std"].append(act_std.item())
                metrics["active_neuron_ratio"].append(active_neurons.item())

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        metrics["loss"].append(avg_loss)
        metrics["accuracy"].append(accuracy)

        pbar.set_postfix(
            {
                "loss": f"{avg_loss:.4f}",
                "acc": f"{accuracy:.1f}%",
                "act%": f'{metrics["active_neuron_ratio"][-1]:.2%}',
            }
        )

    return metrics


def testBasicFunction(brain):
    # Reset brain state
    brain.activations = torch.zeros(brain.neuron_count, device=brain.device)
    brain.activation_history = []
    brain.time_step = 0

    print("\nTesting with input 1.0:")
    input_neurons = brain.input_neurons
    output_neurons = brain.output_neurons

    # Create tensors on the correct device
    all_neurons = torch.arange(brain.neuron_count, device=brain.device)
    io_neurons = torch.cat([input_neurons, output_neurons])
    hidden_neurons = all_neurons[~torch.isin(all_neurons, io_neurons)]

    # Set input
    brain.activations[input_neurons] = 1.0
    print(f"Initial state:")
    print(f"Input neuron activations: {brain.activations[input_neurons].tolist()}")
    print(f"Hidden neuron activations: {brain.activations[hidden_neurons].tolist()}")
    print(f"Output neuron activations: {brain.activations[output_neurons].tolist()}")

    # Step and show each state
    for step in range(3):
        brain.step()
        print(f"\nAfter step {step+1}:")
        print(f"Input neuron activations: {brain.activations[input_neurons].tolist()}")
        print(
            f"Hidden neuron activations: {brain.activations[hidden_neurons].tolist()}"
        )
        print(
            f"Output neuron activations: {brain.activations[output_neurons].tolist()}"
        )


def traceSignalPath(brain):
    from_idx = brain.connection_indices[0]
    to_idx = brain.connection_indices[1]
    weights = brain.connection_weights

    # Count connections per neuron
    inputs_per_neuron = torch.zeros(brain.neuron_count, dtype=torch.int)
    for idx in to_idx:
        inputs_per_neuron[idx] += 1
    outputs_per_neuron = torch.zeros(brain.neuron_count, dtype=torch.int)
    for idx in from_idx:
        outputs_per_neuron[idx] += 1

    print("\nNeuron connectivity:")
    for i in range(brain.neuron_count):
        print(
            f"Neuron {i}: {inputs_per_neuron[i]} inputs, {outputs_per_neuron[i]} outputs"
        )

    print(f"\nTotal connections: {len(weights)}")
    print(f"Average connections per neuron: {len(weights)/brain.neuron_count:.1f}")


def analyzeSinApproximation(brain, n_samples=10):
    # Generate evenly spaced x values
    x = torch.linspace(-np.pi, np.pi, n_samples).unsqueeze(1)
    y = torch.sin(x)

    # Normalize to [0,1] like in training
    x = (x + np.pi) / (2 * np.pi)
    y = (y + 1) / 2

    x = x.to(brain.device)
    y = y.to(brain.device)

    print("\nAnalyzing sin(x) approximation:")
    print("x\t\tTarget\t\tOutput")
    print("-" * 40)
    with torch.no_grad():
        outputs = brain(x)
        for i in range(n_samples):
            print(f"{x[i].item():.3f}\t\t{y[i].item():.3f}\t\t{outputs[i].item():.3f}")


def testInputSensitivity(brain):
    print("\nTesting input sensitivity:")

    # Test with different inputs
    inputs = [-1.0, 0.0, 1.0]

    for input_val in inputs:
        # Reset brain state
        brain.activations = torch.zeros(brain.neuron_count, device=brain.device)
        brain.activation_history = []
        brain.time_step = 0

        # Set input
        x = torch.tensor([[input_val]], device=brain.device)
        with torch.no_grad():
            output = brain(x)

        print(f"\nInput: {input_val}")
        print(f"Output: {output.item():.4f}")


def testSignalFlow(brain):
    print("\nTesting signal flow...")

    # Reset state
    brain.activations = torch.zeros(brain.neuron_count, device=brain.device)

    # Set extreme input values
    brain.activations[: brain.input_size] = torch.tensor([0.0])
    print(f"Input 0.0 -> Output: {brain.activations[-brain.output_size:].item()}")

    brain.step()
    print(f"After step -> Output: {brain.activations[-brain.output_size:].item()}")

    # Reset and test with 1.0
    brain.activations = torch.zeros(brain.neuron_count, device=brain.device)
    brain.activations[: brain.input_size] = torch.tensor([1.0])
    print(f"\nInput 1.0 -> Output: {brain.activations[-brain.output_size:].item()}")

    brain.step()
    print(f"After step -> Output: {brain.activations[-brain.output_size:].item()}")


def testConstantOutput(brain):
    # Input doesn't matter, but need same shape as before
    x = torch.linspace(0, 1, 200).unsqueeze(1)
    # Target is constant 0.5 for all inputs
    y = torch.full_like(x, 0.5)

    x = x.to(brain.device)
    y = y.to(brain.device)

    print(f"\nInitial output neuron biases: {brain.biases[-brain.output_size:].data}")

    metrics = trainNetwork(brain, x, y, n_epochs=5)

    print(f"\nFinal output neuron biases: {brain.biases[-brain.output_size:].data}")

    return metrics, x, y


def testGradientFlow(brain):
    # Set a single input
    x = torch.tensor([[1.0]], device=brain.device)
    y = torch.tensor([[0.5]], device=brain.device)

    # Forward pass with prints
    print("\nForward pass:")
    output = brain(x)
    print(f"Input: {x.item()}")
    print(f"Output: {output.item()}")

    # Compute loss
    loss = (output - y) ** 2
    print(f"Loss: {loss.item()}")

    # Backward pass
    loss.backward()

    # Print gradient info at each step
    print("\nGradient info:")
    for name, param in brain.named_parameters():
        print(f"\n{name}:")
        if param.grad is None:
            print("  No gradient!")
        else:
            print(f"  Gradient shape: {param.grad.shape}")
            print(f"  Gradient values: {param.grad}")
            print(f"  Parameter values: {param.data}")
