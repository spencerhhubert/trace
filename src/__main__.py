from plotting import *
from network import *
from setup import *
from training import *
from testing import *
from baseline import *
from util import *


def main():
    device = setup()

    train_loader, test_loader = getDataLoaders(device)

    brain = Brain(
        device,
        NEURON_COUNT,
        1,
        1,
        True,
    )
    analyzeConnectivity(brain)
    countWeightsAffectingOutputBasedOnInput(brain)
    # testBasicFunction(brain)
    # traceSignalPath(brain)
    # analyzeSinApproximation(brain)
    # testInputSensitivity(brain)
    #


    # two issues
    # 1) can encounter situation where information cannot flow because input points to a neuron that only has inputs
    # 2) actually, this may not be that, it seems like neurons can have connections that go both ways
    metrics, x, y = trainSinX(brain, 20)
    # metrics,x,y = testConstantOutput(brain)
    plotTrainingMetrics(metrics)
    with torch.no_grad():
        y_pred = torch.cat([brain(x[i : i + 10]) for i in range(0, len(x), 10)])
        plotFunctionResults(x, y, y_pred, "Brain: Sin(x) Results")
    animateNetworkActivity(brain)

    return

    # Baseline
    baseline_sin = BaselineMLP(device, input_size=1, output_size=1, hidden_size=20)
    countWeightsAffectingOutput(baseline_sin)
    metrics_baseline, x_baseline, y_baseline = trainBaselineSinX(baseline_sin, n_epochs=500)
    plotTrainingMetrics(metrics_baseline)
    with torch.no_grad():
        y_pred_baseline = baseline_sin(x_baseline)
        plotFunctionResults(
            x_baseline, y_baseline, y_pred_baseline, "Baseline: Sin(x) Results"
        )

    # MNIST
    # train_loader, test_loader = getDataLoaders(device)
    # baseline_mnist = BaselineMLP(device, input_size=28*28, output_size=10)
    # metrics_baseline_mnist = trainBaselineMNIST(baseline_mnist, train_loader)
    # testBaseline(baseline_mnist, test_loader)
    # plotTrainingMetrics(metrics_baseline_mnist)


if __name__ == "__main__":
    main()
