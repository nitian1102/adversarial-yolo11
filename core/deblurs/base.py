import torch
import torch.nn as nn


class DeblurModule(nn.Module):
    """Base class for differentiable deblurring modules."""

    name: str = "deblur"

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def amplification_hint(self, height: int, width: int, device: torch.device) -> torch.Tensor | None:
        """Optional frequency gain map used for mechanism-aware initialization."""
        return None
