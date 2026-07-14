from typing import Any

import torch

from .lightweight_cnn import LightweightUNetDeblur
from .traditional import RichardsonLucyDeblur, WienerDeblur
from .external_models import MPRNetDeblur, NAFNetDeblur


def _resolve_nafnet_width(name: str, kwargs: dict[str, Any]) -> int:
    if name == "nafnet64":
        return 64
    if name == "nafnet32":
        return 32

    weights = str(kwargs.get("weights") or kwargs.get("checkpoint") or "").lower()
    if "width64" in weights:
        return 64
    if "width32" in weights:
        return 32

    width = int(kwargs.get("width") or 32)
    return width if width in {32, 64} else 32


def build_deblur(
    name: str,
    kernel: torch.Tensor,
    device: torch.device | str,
    **kwargs: Any,
):
    name = name.lower()
    if name in {"wiener", "wiener_filter"}:
        return WienerDeblur(kernel=kernel.to(device), balance=kwargs.get("balance", 0.01)).to(device)
    if name in {"rl", "richardson_lucy", "richardson-lucy"}:
        return RichardsonLucyDeblur(
            kernel=kernel.to(device),
            iterations=kwargs.get("iterations", 10),
            eps=kwargs.get("eps", 1e-6),
        ).to(device)
    if name in {"cnn", "unet", "lightweight_unet"}:
        return LightweightUNetDeblur(
            checkpoint=kwargs.get("checkpoint"),
            width=kwargs.get("width", 24),
            residual_scale=kwargs.get("residual_scale", 1.0),
            device=device,
        ).to(device)
    if name in {"mprnet", "mprnet_gopro"}:
        return MPRNetDeblur(
            repo_dir=kwargs.get("repo_dir") or "external/MPRNet",
            weights=kwargs.get("weights") or kwargs.get("checkpoint"),
            device=device,
        ).to(device)
    if name in {"nafnet", "nafnet32", "nafnet64"}:
        width = _resolve_nafnet_width(name, kwargs)
        return NAFNetDeblur(
            repo_dir=kwargs.get("repo_dir") or "external/NAFNet",
            weights=kwargs.get("weights") or kwargs.get("checkpoint"),
            width=width,
            device=device,
        ).to(device)
    raise ValueError(f"Unsupported deblur method: {name}")
