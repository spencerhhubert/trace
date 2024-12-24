import torch

def setup():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
    return device
