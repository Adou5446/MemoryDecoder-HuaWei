import torch


def get_device():
    """Return the best available device: NPU > CUDA > CPU."""
    try:
        import torch_npu
        if torch.npu.is_available():
            return torch.device('npu')
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
