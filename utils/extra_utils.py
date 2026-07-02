import torch
import os
import math
import matplotlib.pyplot as plt

def save_two_tensors_by_channel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    output_path: str,
    label1: str = "tensor1",
    label2: str = "tensor2",
    ncols: int = 1,
):
    """Plot two (1, T, C) tensors on a single multi-channel figure and save it.

    Args:
        x1, x2: shape (1, T, C).
        output_path: target image path, e.g. "/tmp/compare.png".
        label1, label2: legend labels.
        ncols: subplot column count (default 1).
    """
    assert x1.shape == x2.shape, f"shape mismatch: {x1.shape} vs {x2.shape}"
    assert x1.ndim == 3 and x1.shape[0] == 1, f"expected shape (1, T, C), got {x1.shape}"

    x1 = x1.detach().to(torch.float32).cpu()[0]  # (T, C)
    x2 = x2.detach().to(torch.float32).cpu()[0]  # (T, C)

    T, C = x1.shape
    nrows = math.ceil(C / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 3 * nrows),
        squeeze=False
    )
    axes = axes.reshape(-1)

    t = range(T)

    for c in range(C):
        ax = axes[c]
        ax.plot(t, x1[:, c], label=label1)
        ax.plot(t, x2[:, c], label=label2)
        ax.set_title(f"Channel {c}")
        ax.set_xlabel("t")
        ax.set_ylabel("value")
        ax.legend()
        ax.grid(True)

    for i in range(C, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
