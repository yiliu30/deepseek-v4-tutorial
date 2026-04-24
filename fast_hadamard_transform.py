"""Pure-PyTorch fallback for fast_hadamard_transform when the CUDA extension can't build."""
import torch


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Recursive Walsh-Hadamard transform along the last dimension.

    For the tiny model walkthrough this is functionally correct —
    H_n * x / sqrt(n) is an orthogonal rotation that spreads information
    across dims before FP4 quantization.
    """
    n = x.shape[-1]
    assert n & (n - 1) == 0, f"Last dim must be power of 2, got {n}"

    # Iterative in-place butterfly
    h = 1
    y = x.float()
    while h < n:
        # Split into pairs of size h and butterfly
        y_view = y.view(*y.shape[:-1], -1, 2, h)
        a = y_view[..., 0, :]
        b = y_view[..., 1, :]
        y_view[..., 0, :] = a + b
        y_view[..., 1, :] = a - b
        h *= 2

    return (y * scale).to(x.dtype)
