from collections import OrderedDict
from pathlib import Path
from runpy import run_path
import sys

import torch
import torch.nn.functional as F

from .base import DeblurModule


class MPRNetDeblur(DeblurModule):
    """Official MPRNet deblurring model wrapper.

    Expected repo layout:
        external/MPRNet/Deblurring/MPRNet.py

    Expected pretrained weights:
        external/MPRNet/Deblurring/pretrained_models/model_deblurring.pth
    """

    name = "mprnet"

    def __init__(
        self,
        repo_dir: str = "external/MPRNet",
        weights: str | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.repo_dir = Path(repo_dir)
        model_file = self.repo_dir / "Deblurring" / "MPRNet.py"
        if not model_file.exists():
            raise FileNotFoundError(
                f"MPRNet repo not found at {self.repo_dir}. "
                "Run: git clone --depth 1 https://github.com/swz30/MPRNet.git external/MPRNet"
            )
        weights_path = Path(weights) if weights else self.repo_dir / "Deblurring" / "pretrained_models" / "model_deblurring.pth"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"MPRNet deblurring weights not found: {weights_path}\n"
                "Run: python scripts/download_deblur_weights.py --model mprnet\n"
                "or manually download the official MPRNet Deblurring model and place it there."
            )

        load_file = run_path(str(model_file))
        self.model = load_file["MPRNet"]()
        self._load_checkpoint(self.model, weights_path)
        self.model.eval()
        if device is not None:
            self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _load_checkpoint(model: torch.nn.Module, weights: Path) -> None:
        checkpoint = torch.load(str(weights), map_location="cpu")
        state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        try:
            model.load_state_dict(state)
        except RuntimeError:
            new_state = OrderedDict()
            for key, value in state.items():
                new_state[key[7:] if key.startswith("module.") else key] = value
            model.load_state_dict(new_state)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        multiple = 8
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else image
        restored = self.model(padded)
        if isinstance(restored, (list, tuple)):
            restored = restored[0]
        restored = restored[:, :, :h, :w]
        return restored.clamp(0.0, 1.0)


class NAFNetDeblur(DeblurModule):
    """Official NAFNet GoPro deblurring model wrapper.

    Expected repo layout:
        external/NAFNet/basicsr/models/archs/NAFNet_arch.py

    Expected pretrained weights:
        external/NAFNet/experiments/pretrained_models/NAFNet-GoPro-width32.pth
        or NAFNet-GoPro-width64.pth
    """

    name = "nafnet"

    def __init__(
        self,
        repo_dir: str = "external/NAFNet",
        weights: str | None = None,
        width: int = 32,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.repo_dir = Path(repo_dir)
        arch_file = self.repo_dir / "basicsr" / "models" / "archs" / "NAFNet_arch.py"
        if not arch_file.exists():
            raise FileNotFoundError(
                f"NAFNet repo not found at {self.repo_dir}. "
                "Run: git clone --depth 1 https://github.com/megvii-research/NAFNet.git external/NAFNet"
            )
        weights_path = (
            Path(weights)
            if weights
            else self.repo_dir / "experiments" / "pretrained_models" / f"NAFNet-GoPro-width{width}.pth"
        )
        if not weights_path.exists():
            raise FileNotFoundError(
                f"NAFNet weights not found: {weights_path}\n"
                "Run: python scripts/download_deblur_weights.py --model nafnet32\n"
                "or manually download the official NAFNet GoPro weight and place it there."
            )

        repo_str = str(self.repo_dir.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        from basicsr.models.archs.NAFNet_arch import NAFNet  # type: ignore

        self.model = NAFNet(
            img_channel=3,
            width=int(width),
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 28],
            dec_blk_nums=[1, 1, 1, 1],
        )
        self._load_checkpoint(self.model, weights_path)
        self.model.eval()
        if device is not None:
            self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _load_checkpoint(model: torch.nn.Module, weights: Path) -> None:
        checkpoint = torch.load(str(weights), map_location="cpu")
        if isinstance(checkpoint, dict):
            if "params_ema" in checkpoint:
                state = checkpoint["params_ema"]
            elif "params" in checkpoint:
                state = checkpoint["params"]
            elif "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
            else:
                state = checkpoint
        else:
            state = checkpoint
        new_state = OrderedDict()
        for key, value in state.items():
            key = key[7:] if key.startswith("module.") else key
            key = key[6:] if key.startswith("net_g.") else key
            new_state[key] = value
        model.load_state_dict(new_state)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image).clamp(0.0, 1.0)
