from plotting import *
from baseline import *
from brain import *
import os


def main():
    device = "cpu"
    model_path = "brain.pt"

    if os.path.exists(model_path):
        print("Loading pre-trained brain model...")
        brain = Brain.load(model_path, device)
    else:
        print("Training new brain model...")
        brain = Brain(device, input_size=1, output_size=1, init_strategy="spatial")
        # metrics_brain, x_brain, y_brain = trainLinear(brain, n_epochs=10, lr_val=0.01)
        metrics_brain, x_brain, y_brain = trainSinX(brain, n_epochs=10, lr_val=0.01)
        # metrics_brain, x_brain, y_brain = trainPolynomial(brain, n_epochs=200, lr_val=0.01)
        plotTrainingMetrics(metrics_brain)
        y_pred_brain = torch.zeros_like(y_brain)
        for i in range(len(x_brain)):
            x_i = x_brain[i : i + 1]
            y_pred_brain[i] = brain(x_i)
        plotFunctionResults(x_brain, y_brain, y_pred_brain, "Brain: Linear Results")

        brain.save(model_path)

    visualizeBrain(brain, show_weights=False, weight_thickness=True)
    with torch.no_grad():
        test_input = torch.tensor([[0.5]], device=device)
        brain(test_input)
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
