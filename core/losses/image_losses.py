import torch
import torch.nn.functional as F


def normalize_map(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = x.flatten(start_dim=1)
    mn = flat.min(dim=1).values.view(-1, 1, 1, 1)
    mx = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (x - mn) / (mx - mn + eps)


def edge_map(image: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=1, keepdim=True)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kx)
    gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), ky)
    return torch.sqrt(gx.square() + gy.square() + 1e-12)


def edge_difference(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    diff = (edge_map(a) - edge_map(b)).abs()
    if mask is not None:
        diff = diff * mask
        return diff.sum() / mask.sum().clamp_min(1.0)
    return diff.mean()


def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w
