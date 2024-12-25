import torch
from torch import nn
from tqdm import tqdm

def trainNetwork(brain, x, y, n_epochs=1000, batch_size=10, learning_rate=0.01):
    optimizer = torch.optim.Adam(brain.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    metrics = {
        'loss': [],
        'grad_norm': [],
        'activation_mean': [],
        'activation_std': [],
        'active_neuron_ratio': []
    }

    pbar = tqdm(range(n_epochs), desc='Training')
    for epoch in pbar:
        optimizer.zero_grad()

        total_loss = 0
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            output = brain(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            total_loss += loss.item()

            # Collect debug metrics
            with torch.no_grad():
                # Gradient norm across all parameters
                grad_norm = torch.norm(torch.stack([
                    p.grad.norm()
                    for p in brain.parameters()
                    if p.grad is not None
                ]))

                # Activation statistics
                active_neurons = (brain.activations.abs() > 0.01).float().mean()
                act_mean = brain.activations.mean()
                act_std = brain.activations.std()

                metrics['grad_norm'].append(grad_norm.item())
                metrics['activation_mean'].append(act_mean.item())
                metrics['activation_std'].append(act_std.item())
                metrics['active_neuron_ratio'].append(active_neurons.item())

        optimizer.step()

        avg_loss = total_loss / (len(x) // batch_size)
        metrics['loss'].append(avg_loss)

        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'grad': f'{metrics["grad_norm"][-1]:.3f}',
            'act%': f'{metrics["active_neuron_ratio"][-1]:.2%}'
        })

    return metrics
