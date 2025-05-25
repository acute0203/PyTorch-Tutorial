import torch
import platform

def get_device():
    os_name = platform.system()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA on {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # MPS 適用於 macOS + Apple Silicon
        if os_name == "Darwin":
            device = torch.device("mps")
            print("Using Apple Silicon MPS backend")
        else:
            device = torch.device("cpu")
            print("MPS backend found but not using macOS, fallback to CPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device

# 使用方式
device = get_device()

# 建立範例 tensor
x = torch.randn(3, 3).to(device)
print(f"Tensor is on: {x.device}")
