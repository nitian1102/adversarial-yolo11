import torch
import torch.nn.functional as F

from .base import DeblurModule


def _psf_to_otf(kernel_2d: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert a spatial PSF kernel to an OTF with the given image size."""
    psf = torch.zeros((height, width), device=kernel_2d.device, dtype=kernel_2d.dtype)
    kh, kw = kernel_2d.shape
    psf[:kh, :kw] = kernel_2d
    psf = torch.roll(psf, shifts=(-(kh // 2), -(kw // 2)), dims=(0, 1))
    return torch.fft.fft2(psf)


class WienerDeblur(DeblurModule):
    """Differentiable Wiener deconvolution baseline.

    This baseline models a classical deterministic deblur method. It exposes
    the inverse-filter gain map so attacks can initialize residuals in
    frequencies that the restoration algorithm tends to amplify.
    """

    name = "wiener"

    def __init__(self, kernel: torch.Tensor, balance: float = 0.01):
        super().__init__()
        if kernel.ndim != 4:
            raise ValueError("Kernel must be [C,1,H,W]")
        self.register_buffer("kernel", kernel.detach().clone())
        self.balance = float(balance)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b, c, h, w = image.shape
        outputs = []
        kernel = self.kernel.to(device=image.device, dtype=image.dtype)
        for channel in range(c):
            k2d = kernel[min(channel, kernel.shape[0] - 1), 0]
            otf = _psf_to_otf(k2d, h, w)
            gain = torch.conj(otf) / (torch.abs(otf) ** 2 + self.balance)
            image_fft = torch.fft.fft2(image[:, channel])
            restored = torch.fft.ifft2(image_fft * gain).real
            outputs.append(restored.unsqueeze(1))
        return torch.cat(outputs, dim=1).clamp(0.0, 1.0)

    def amplification_hint(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        kernel = self.kernel.to(device=device)
        k2d = kernel[0, 0]
        otf = _psf_to_otf(k2d, height, width)
        gain = torch.abs(torch.conj(otf) / (torch.abs(otf) ** 2 + self.balance))
        gain = gain / gain.max().clamp_min(1e-12)
        return gain


class RichardsonLucyDeblur(DeblurModule):
    """Differentiable Richardson-Lucy deconvolution baseline."""

    name = "richardson_lucy"

    def __init__(self, kernel: torch.Tensor, iterations: int = 10, eps: float = 1e-6):
        super().__init__()
        if kernel.ndim != 4:
            raise ValueError("Kernel must be [C,1,H,W]")
        self.register_buffer("kernel", kernel.detach().clone())
        self.iterations = int(iterations)
        self.eps = float(eps)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        channels = image.shape[1]
        kernel = self.kernel.to(device=image.device, dtype=image.dtype)
        if kernel.shape[0] != channels:
            kernel = kernel[:1].repeat(channels, 1, 1, 1)
        flipped = torch.flip(kernel, dims=(-2, -1))
        estimate = image.clamp_min(self.eps)
        pad_y = kernel.shape[-2] // 2
        pad_x = kernel.shape[-1] // 2
        for _ in range(self.iterations):
            blurred_est = F.conv2d(
                F.pad(estimate, (pad_x, pad_x, pad_y, pad_y), mode="reflect"),
                kernel,
                groups=channels,
            ).clamp_min(self.eps)
            ratio = image / blurred_est
            correction = F.conv2d(
                F.pad(ratio, (pad_x, pad_x, pad_y, pad_y), mode="reflect"),
                flipped,
                groups=channels,
            )
            estimate = (estimate * correction).clamp(self.eps, 1.0)
        return estimate.clamp(0.0, 1.0)
