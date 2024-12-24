import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from plotting import *
from network import *
from setup import *
from training import *
from testing import *

def demo():
    device = setup()
    brain = Brain(device)

    # First try identity function
    metrics_identity, x_id, y_id = trainIdentity(brain)
    plotTrainingMetrics(metrics_identity)

    # Then try sin(x)
    metrics_sin, x_sin, y_sin = trainSinX(brain)
    plotTrainingMetrics(metrics_sin)
    plotSinXResults(brain, x_sin, y_sin)

if __name__ == "__main__":
    demo()
