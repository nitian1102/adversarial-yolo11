from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass
class MotionBlurConfig:
    kernel_size: int = 21
    angle: float = 0.0
    channels: int = 3


def _normalize_kernel(kernel: torch.Tensor) -> torch.Tensor:
    denom = kernel.sum().clamp_min(1e-12)
    return kernel / denom


def build_motion_kernel(
    kernel_size: int = 21,
    angle: float = 0.0,
    channels: int = 3,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a depthwise 2D linear motion blur kernel.

    Returns a tensor shaped [channels, 1, kernel_size, kernel_size].
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 3:
        kernel_size = 3

    center = (kernel_size - 1) / 2.0
    theta = math.radians(angle)
    direction = torch.tensor(
        [math.cos(theta), math.sin(theta)], device=device, dtype=dtype
    )

    kernel = torch.zeros((kernel_size, kernel_size), device=device, dtype=dtype)
    samples = torch.linspace(-center, center, steps=kernel_size, device=device, dtype=dtype)
    xs = center + samples * direction[0]
    ys = center + samples * direction[1]

    # Bilinear splat keeps the kernel stable for non-axis-aligned angles.
    for x, y in zip(xs, ys):
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        for dx in (0, 1):
            for dy in (0, 1):
                xi = x0 + dx
                yi = y0 + dy
                if 0 <= xi < kernel_size and 0 <= yi < kernel_size:
                    wx = 1.0 - torch.abs(x - xi.to(dtype))
                    wy = 1.0 - torch.abs(y - yi.to(dtype))
                    kernel[yi, xi] += torch.clamp(wx, min=0.0) * torch.clamp(wy, min=0.0)

    kernel = _normalize_kernel(kernel)
    return kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


def apply_motion_blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply depthwise motion blur to a BCHW tensor in [0, 1]."""
    if image.ndim != 4:
        raise ValueError(f"Expected BCHW image tensor, got shape {tuple(image.shape)}")
    if kernel.ndim != 4:
        raise ValueError(f"Expected kernel [C,1,H,W], got shape {tuple(kernel.shape)}")
    channels = image.shape[1]
    if kernel.shape[0] != channels:
        if kernel.shape[0] == 1:
            kernel = kernel.repeat(channels, 1, 1, 1)
        else:
            raise ValueError(f"Kernel channels {kernel.shape[0]} do not match image {channels}")
    pad_y = kernel.shape[-2] // 2
    pad_x = kernel.shape[-1] // 2
    padded = F.pad(image, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
    return F.conv2d(padded, kernel.to(device=image.device, dtype=image.dtype), groups=channels)
