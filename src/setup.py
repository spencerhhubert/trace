import torch


def setup():
    # device = (
    #     "mps"
    #     if torch.backends.mps.is_available()
    #     else "cuda"
    #     if torch.cuda.is_available()
    #     else "cpu"
    # )
    device = "cpu"
    # cpu seems faster on m2? like 2x faster for regular mlp
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
    return device
