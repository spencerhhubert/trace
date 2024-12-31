from plotting import *
from setup import *
from baseline import *


def main():
    device = setup()

    # Baseline
    baseline_sin = BaselineMLP(device, input_size=1, output_size=1, hidden_size=5)
    countWeightsAffectingBaselineOutput(baseline_sin)
    metrics_baseline, x_baseline, y_baseline = trainBaselineSinX(
        baseline_sin, n_epochs=1000
    )
    plotTrainingMetrics(metrics_baseline)
    with torch.no_grad():
        y_pred_baseline = baseline_sin(x_baseline)
        plotFunctionResults(
            x_baseline, y_baseline, y_pred_baseline, "Baseline: Sin(x) Results"
        )


if __name__ == "__main__":
    main()
