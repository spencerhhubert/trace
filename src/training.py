import torch
from torch import nn
from tqdm import tqdm
import numpy as np

def trainNetwork(brain, x, y, n_epochs=1000, batch_size=10, learning_rate=0.01):
    optimizer = torch.optim.Adam(brain.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Print initial state
    print("\nInitial network state:")
    brain.debugState()

    metrics = {
        'loss': [],
        'predictions': [],
        'weight_means': [],  # track weight statistics
        'weight_stds': []
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

            # Debug first batch of first epoch
            if epoch == 0 and i == 0:
                print(f"\nFirst batch debug:")
                print(f"Input range: {batch_x.min():.3f} to {batch_x.max():.3f}")
                print(f"Output range: {output.min():.3f} to {output.max():.3f}")
                print(f"Target range: {batch_y.min():.3f} to {batch_y.max():.3f}")
                print(f"Initial loss: {loss.item():.5f}")

                # Check gradients
                grad_norms = [p.grad.norm().item() for p in brain.parameters() if p.grad is not None]
                print(f"Gradient norms: {grad_norms}")

        optimizer.step()

        avg_loss = total_loss / (len(x) // batch_size)
        metrics['loss'].append(avg_loss)

        # Track weight statistics
        metrics['weight_means'].append(brain.connection_weights.mean().item())
        metrics['weight_stds'].append(brain.connection_weights.std().item())

        if epoch % 100 == 0:
            with torch.no_grad():
                test_x = x[:batch_size]
                pred_y = brain(test_x)
                metrics['predictions'].append(pred_y.cpu().numpy())
                brain.debugState()

        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'w_mean': f'{metrics["weight_means"][-1]:.3f}'
        })

    print("\nFinal network state:")
    brain.debugState()
    return metrics
