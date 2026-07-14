from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DeblurModule


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleDeblurUNet(nn.Module):
    """A compact residual UNet for synthetic motion-deblur experiments."""

    def __init__(self, channels: int = 3, width: int = 24, residual_scale: float = 1.0):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.enc1 = _ConvBlock(channels, width)
        self.enc2 = _ConvBlock(width, width * 2)
        self.enc3 = _ConvBlock(width * 2, width * 4)
        self.dec2 = _ConvBlock(width * 4 + width * 2, width * 2)
        self.dec1 = _ConvBlock(width * 2 + width, width)
        self.out = nn.Conv2d(width, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        d2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        residual = torch.tanh(self.out(d1))
        return (x + self.residual_scale * residual).clamp(0.0, 1.0)


class LightweightUNetDeblur(DeblurModule):
    """Wrapper for a lightweight CNN deblur baseline.

    Use scripts/train_lightweight_deblur.py to train a checkpoint on synthetic
    motion-blur pairs. If no checkpoint is provided the model is untrained and
    should only be used for smoke tests.
    """

    name = "lightweight_unet"

    def __init__(
        self,
        checkpoint: str | None = None,
        width: int = 24,
        residual_scale: float = 1.0,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        state = None
        if checkpoint:
            ckpt_path = Path(checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Deblur checkpoint not found: {checkpoint}")
            state = torch.load(str(ckpt_path), map_location="cpu")
            if isinstance(state, dict) and "width" in state:
                width = int(state["width"])
            elif isinstance(state, dict):
                weights = state["model"] if "model" in state else state
                first = weights.get("enc1.net.0.weight") if isinstance(weights, dict) else None
                if first is not None:
                    width = int(first.shape[0])
        self.model = SimpleDeblurUNet(width=width, residual_scale=residual_scale)
        if state is not None:
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state)
        if device is not None:
            self.model.to(device)
        self.model.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)
