import matplotlib.pyplot as plt


def plotTrainingMetrics(metrics):
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 6))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, metrics.items()):
        ax.plot(values)
        ax.set_title(name)
        ax.set_xlabel("Step")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plotFunctionResults(x, y_true, y_pred, title="Function Comparison"):
    plt.figure(figsize=(10, 5))
    plt.plot(x.cpu().numpy(), y_true.cpu().numpy(), label="True function")
    plt.plot(x.cpu().numpy(), y_pred.cpu().numpy(), "--", label="Model output")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()
