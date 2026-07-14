import argparse
import glob
import json
from pathlib import Path
import sys
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont
import torch
import yaml
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.attacks.deblur_amplified import (  # noqa: E402
    DeblurAmplifiedAttackConfig,
    EnsembleDeblurAmplifiedAttack,
)
from core.deblurs import build_deblur  # noqa: E402
from core.detectors import YOLOv5Detector  # noqa: E402
from core.transforms import apply_motion_blur, build_motion_kernel  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEBLUR_ALIASES = {
    "wiener": "wiener",
    "rl": "richardson_lucy",
    "richardson_lucy": "richardson_lucy",
    "mprnet": "mprnet",
    "nafnet": "nafnet",
    "nafnet32": "nafnet32",
    "nafnet64": "nafnet64",
    "cnn": "lightweight_unet",
    "unet": "lightweight_unet",
    "lightweight_unet": "lightweight_unet",
}


def load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def collect_images(patterns: list[str]) -> list[Path]:
    if not patterns:
        raise FileNotFoundError("No images configured. Pass --images <file/folder/glob>.")
    paths: list[Path] = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if expanded:
            paths.extend(Path(p) for p in expanded)
            continue
        path = Path(pattern)
        if path.is_dir():
            paths.extend(x for x in path.rglob("*") if x.suffix.lower() in IMAGE_EXTS)
        elif path.exists() and path.suffix.lower() in IMAGE_EXTS:
            paths.append(path)
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    if not unique:
        raise FileNotFoundError(f"No images matched: {patterns}")
    return unique


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]
    return TF.to_pil_image(tensor)


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(path)


def draw_detections(img: Image.Image, detections: list[dict]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for det in detections:
        box = det["box"]
        label = f'{det["label"]} {det["score"]:.2f}'
        draw.rectangle(box, outline=(255, 40, 40), width=2)
        text_box = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle(text_box, fill=(255, 40, 40))
        draw.text((box[0], box[1]), label, fill=(255, 255, 255), font=font)
    return out


def make_panel(items: list[tuple[str, Image.Image]], output: Path) -> None:
    thumb_w = 360
    thumb_h = 260
    pad = 24
    title_h = 30
    panel = Image.new("RGB", (len(items) * (thumb_w + pad) + pad, thumb_h + title_h + 2 * pad), "white")
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for i, (title, img) in enumerate(items):
        x = pad + i * (thumb_w + pad)
        y = pad + title_h
        resized = img.copy()
        resized.thumbnail((thumb_w, thumb_h))
        bg = Image.new("RGB", (thumb_w, thumb_h), (245, 245, 245))
        bg.paste(resized, ((thumb_w - resized.width) // 2, (thumb_h - resized.height) // 2))
        panel.paste(bg, (x, y))
        draw.text((x, pad), title, fill=(0, 0, 0), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)


def load_image(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, [image_size, image_size])
    return TF.to_tensor(img).unsqueeze(0).to(device)


def condition_summary(detector: YOLOv5Detector, name: str, image: torch.Tensor) -> dict:
    pred = detector.raw(image)
    summary = detector.summarize_raw(pred)
    return {
        "condition": name,
        "count": summary.count,
        "mean_confidence": summary.mean_confidence,
        "max_confidence": summary.max_confidence,
    }


def split_methods(values: list[str]) -> list[str]:
    methods = []
    for value in values:
        for token in value.split(","):
            token = token.strip().lower()
            if not token:
                continue
            if token not in DEBLUR_ALIASES:
                raise ValueError(f"Unknown deblur method: {token}")
            method = DEBLUR_ALIASES[token]
            if method not in methods:
                methods.append(method)
    return methods


def normalize_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def first_existing(paths: list[str | Path | None]) -> Path | None:
    for path in paths:
        candidate = normalize_path(path)
        if candidate and candidate.exists():
            return candidate
    return None


def method_kwargs(method: str, base_cfg: dict, args: argparse.Namespace) -> dict | None:
    kwargs = {k: v for k, v in base_cfg.items() if k != "method"}
    kwargs["checkpoint"] = None
    kwargs["weights"] = None
    kwargs["repo_dir"] = None

    if method == "lightweight_unet":
        checkpoint = first_existing([args.cnn_checkpoint, ROOT / "output" / "deblur" / "lightweight_unet.pt"])
        if checkpoint is None:
            print("[skip] lightweight_unet has no checkpoint.", flush=True)
            return None
        kwargs["checkpoint"] = str(checkpoint)
    elif method == "mprnet":
        weights = first_existing(
            [
                args.mprnet_weights,
                ROOT / "weights" / "model_deblurring.pth",
                ROOT / "external" / "MPRNet" / "Deblurring" / "pretrained_models" / "model_deblurring.pth",
            ]
        )
        if weights is None:
            print("[skip] mprnet has no official deblurring weights.", flush=True)
            return None
        kwargs["checkpoint"] = str(weights)
    elif method in {"nafnet", "nafnet32", "nafnet64"}:
        width = 64 if method == "nafnet64" else 32
        weights = first_existing(
            [
                args.nafnet64_weights if width == 64 else args.nafnet32_weights,
                args.nafnet_weights,
                ROOT / "weights" / f"NAFNet-GoPro-width{width}.pth",
                ROOT / "external" / "NAFNet" / "experiments" / "pretrained_models" / f"NAFNet-GoPro-width{width}.pth",
            ]
        )
        if weights is None:
            print(f"[skip] {method} has no official NAFNet width{width} weights.", flush=True)
            return None
        kwargs["checkpoint"] = str(weights)
        kwargs["width"] = width
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one shared perturbation against multiple deblur methods.")
    parser.add_argument("--config", default="configs/deblur_amplified.yaml")
    parser.add_argument("--images", nargs="*", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deblurs", nargs="+", required=True)
    parser.add_argument("--cnn-checkpoint")
    parser.add_argument("--mprnet-weights")
    parser.add_argument("--nafnet-weights")
    parser.add_argument("--nafnet32-weights")
    parser.add_argument("--nafnet64-weights")
    parser.add_argument("--device")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--kernel-size", type=int)
    parser.add_argument("--angle", type=float)
    parser.add_argument("--mask", choices=["on", "off"])
    args = parser.parse_args()

    defaults = {
        "paths": {"output_dir": "output/deblur_ensemble", "images": []},
        "runtime": {"device": "auto", "image_size": 640},
        "motion_blur": {"kernel_size": 21, "angle": 0.0},
        "detector": {
            "model_name": "yolov5s",
            "weights": None,
            "repo_or_dir": "ultralytics/yolov5",
            "source": "github",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
        },
        "deblur": {
            "method": "wiener",
            "balance": 0.01,
            "iterations": 10,
            "checkpoint": None,
            "weights": None,
            "repo_dir": None,
            "width": 24,
            "residual_scale": 1.0,
        },
        "attack": {},
    }
    cfg = merge_dict(defaults, load_yaml(args.config))
    cfg["paths"]["images"] = args.images
    cfg["paths"]["output_dir"] = args.output_dir
    if args.device:
        cfg["runtime"]["device"] = args.device
    if args.image_size:
        cfg["runtime"]["image_size"] = args.image_size
    if args.steps is not None:
        cfg["attack"]["steps"] = args.steps
    if args.epsilon is not None:
        cfg["attack"]["epsilon"] = args.epsilon
    if args.topk is not None:
        cfg["attack"]["topk"] = args.topk
    if args.kernel_size is not None:
        cfg["motion_blur"]["kernel_size"] = args.kernel_size
    if args.angle is not None:
        cfg["motion_blur"]["angle"] = args.angle
    if args.mask is not None:
        cfg["attack"]["use_region_mask"] = args.mask == "on"

    image_paths = collect_images(cfg["paths"]["images"])
    output_root = Path(cfg["paths"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    device_name = cfg["runtime"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    image_size = int(cfg["runtime"]["image_size"])

    kernel = build_motion_kernel(
        kernel_size=int(cfg["motion_blur"]["kernel_size"]),
        angle=float(cfg["motion_blur"]["angle"]),
        channels=3,
        device=device,
    )
    detector = YOLOv5Detector(device=device, **cfg["detector"])

    methods = split_methods(args.deblurs)
    deblur_models = {}
    for method in methods:
        kwargs = method_kwargs(method, cfg["deblur"], args)
        if kwargs is None:
            continue
        deblur_models[method] = build_deblur(method, kernel=kernel, device=device, **kwargs)
    if not deblur_models:
        raise RuntimeError("No deblur models were available for the ensemble attack.")

    attack_config = DeblurAmplifiedAttackConfig(**cfg.get("attack", {}))
    attack = EnsembleDeblurAmplifiedAttack(detector, deblur_models, attack_config)

    resolved_record = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "entry_command": " ".join(sys.argv),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "python": sys.executable,
        "attack_mode": "ensemble",
        "requested_deblurs": methods,
        "resolved_deblurs": list(deblur_models.keys()),
        "resolved_config": cfg,
        "resolved_images": [str(p) for p in image_paths],
        "device": str(device),
        "image_size": image_size,
    }
    with open(output_root / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved_record, f, indent=2, ensure_ascii=False)

    all_metrics = []
    for i, path in enumerate(image_paths):
        print(f"[{i + 1}/{len(image_paths)}] {path}")
        clean = load_image(path, image_size=image_size, device=device)
        blurred = apply_motion_blur(clean, kernel)
        result = attack.run(blurred)
        adv_blurred = result["adv_blurred"]
        delta = result["delta"]
        mask = result["mask"]
        per_deblur = result["per_deblur"]

        sample_dir = output_root / f"{i:04d}_{path.stem}"
        save_tensor(sample_dir / "B0_clean.png", clean)
        save_tensor(sample_dir / "B1_blur.png", blurred)
        save_tensor(sample_dir / "B4_adv_blur.png", adv_blurred)
        save_tensor(sample_dir / "delta_abs.png", delta.abs() / attack_config.epsilon)
        save_tensor(sample_dir / "region_mask.png", mask.repeat(1, 3, 1, 1))

        det_blur = draw_detections(tensor_to_pil(blurred), detector.detections(blurred))
        det_adv_blur = draw_detections(tensor_to_pil(adv_blurred), detector.detections(adv_blurred))
        sample_metrics = {
            "image": str(path),
            "attack_mode": "ensemble",
            "motion_blur": cfg["motion_blur"],
            "aggregate_attack": result["metrics"],
            "deblurs": {},
        }

        for method, item in per_deblur.items():
            deblurred = item["deblurred"]
            adv_deblurred = item["adv_deblurred"]
            method_dir = sample_dir / method
            save_tensor(method_dir / "B2_deblur.png", deblurred)
            save_tensor(method_dir / "B5_adv_deblur.png", adv_deblurred)
            det_deblur = draw_detections(tensor_to_pil(deblurred), detector.detections(deblurred))
            det_adv_deblur = draw_detections(tensor_to_pil(adv_deblurred), detector.detections(adv_deblurred))
            det_deblur.save(method_dir / "B2_deblur_detection.png")
            det_adv_deblur.save(method_dir / "B5_adv_deblur_detection.png")
            make_panel(
                [
                    ("B0 clean", tensor_to_pil(clean)),
                    ("B1 blur", det_blur),
                    (f"B2 {method}", det_deblur),
                    ("B4 shared adv blur", det_adv_blur),
                    (f"B5 {method}", det_adv_deblur),
                ],
                method_dir / "panel.png",
            )

            conditions = [
                condition_summary(detector, "B0_clean", clean),
                condition_summary(detector, "B1_blur", blurred),
                condition_summary(detector, "B4_shared_adv_blur", adv_blurred),
                condition_summary(detector, f"B2_{method}_deblur", deblurred),
                condition_summary(detector, f"B5_{method}_adv_deblur", adv_deblurred),
            ]
            metrics = {
                "image": str(path),
                "attack_mode": "ensemble",
                "deblur_method": method,
                "motion_blur": cfg["motion_blur"],
                "attack": item["metrics"],
                "aggregate_attack": result["metrics"],
                "conditions": conditions,
            }
            with open(method_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            sample_metrics["deblurs"][method] = metrics
            print(method, json.dumps(item["metrics"], indent=2))

        with open(sample_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(sample_metrics, f, indent=2, ensure_ascii=False)
        all_metrics.append(sample_metrics)

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"saved ensemble outputs: {output_root}")


if __name__ == "__main__":
    main()
