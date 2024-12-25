from plotting import *
from network import *
from setup import *
from training import *
from testing import *
from baseline import *
from util import *


def main():
    device = setup()

    # # Brain stuff
    # brain = Brain(device)
    # metrics_identity = trainIdentity(brain)
    # plotTrainingMetrics(metrics_identity)
    #
    train_loader, test_loader = getDataLoaders(device)

    brain = Brain(device)
    metrics = trainSinX(brain)
    plotTrainingMetrics(metrics)
    visualizeNetwork(brain)
    # metrics = trainMNIST(brain, train_loader, n_epochs=100)

    # Baseline Identity
    # baseline_identity = BaselineMLP(device, input_size=1, output_size=1)
    # metrics_baseline_identity = trainBaselineIdentity(baseline_identity)
    # plotTrainingMetrics(metrics_baseline_identity)

    # # Baseline Sin(x)
    # baseline_sin = BaselineMLP(device, input_size=1, output_size=1)
    # metrics_baseline_sin = trainBaselineSinX(baseline_sin)
    # plotTrainingMetrics(metrics_baseline_sin)

    # MNIST
    # train_loader, test_loader = getDataLoaders(device)
    # baseline_mnist = BaselineMLP(device, input_size=28*28, output_size=10)
    # metrics_baseline_mnist = trainBaselineMNIST(baseline_mnist, train_loader)
    # testBaseline(baseline_mnist, test_loader)
    # plotTrainingMetrics(metrics_baseline_mnist)


if __name__ == "__main__":
    main()
