from setup import *
from plotting import *
from baseline import *
from brain import *


def main():
    device = setup()

    brain = Brain(
        device, input_size=1, output_size=1, init_strategy="spatial"
    )  # or "mlp"
    # metrics_brain, x_brain, y_brain = trainLinear(brain, n_epochs=10, lr_val=0.01)
    metrics_brain, x_brain, y_brain = trainSinX(brain, n_epochs=10, lr_val=0.01)
    # metrics_brain, x_brain, y_brain = trainPolynomial(brain, n_epochs=200, lr_val=0.01)
    plotTrainingMetrics(metrics_brain)
    visualizeBrain(brain, show_weights=True, weight_thickness=True)
    with torch.no_grad():
        y_pred_brain = torch.zeros_like(y_brain)
        for i in range(len(x_brain)):
            x_i = x_brain[i : i + 1]
            y_pred_brain[i] = brain(x_i)
        plotFunctionResults(x_brain, y_brain, y_pred_brain, "Brain: Linear Results")
        animateBrain(brain)

    return

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
