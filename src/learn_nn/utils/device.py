
import torch

_CACHED_DEVICE = None
def get_cached_device():
    global _CACHED_DEVICE
    if _CACHED_DEVICE is not None:
        return _CACHED_DEVICE
    if torch.cuda.is_available():
        _CACHED_DEVICE = torch.device("cuda")
    elif torch.mps.is_available():
        _CACHED_DEVICE = torch.device("mps")
    else:
        _CACHED_DEVICE = torch.device("cpu")

    print(f"cache device is: {_CACHED_DEVICE}")
    return _CACHED_DEVICE
