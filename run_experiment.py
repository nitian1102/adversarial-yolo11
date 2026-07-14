import argparse
from datetime import datetime
import glob
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GLOB_CHARS = {"*", "?", "["}
SKIP_IMAGE_DIRS = {"output", "tmp", ".git", "__pycache__"}

PRESETS = {
    "quick": {"image_size": 160, "steps": 1, "topk": 10, "kernel_size": 9},
    "dev": {"image_size": 320, "steps": 5, "topk": 20, "kernel_size": 15},
    "formal": {"image_size": 640, "steps": 20, "topk": 40, "kernel_size": 9},
}
# python scripts/download_deblur_weights.py --model mprnet

DEBLUR_ALIASES = {
    "wiener": "wiener",
    "rl": "richardson_lucy",
    "mprnet": "mprnet",
    "nafnet": "nafnet",
    "nafnet32": "nafnet32",
    "nafnet64": "nafnet64",
    "cnn": "lightweight_unet",
    "unet": "lightweight_unet",
    "lightweight_unet": "lightweight_unet",
}

DEBLUR_DISPLAY_NAMES = {
    "wiener": "wiener",
    "richardson_lucy": "rl",
    "mprnet": "mprnet",
    "nafnet": "nafnet",
    "nafnet32": "nafnet32",
    "nafnet64": "nafnet64",
    "lightweight_unet": "unet",
}


def is_workspace_image(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
        return False
    return not any(part in SKIP_IMAGE_DIRS for part in path.parts)


def iter_workspace_images() -> list[Path]:
    preferred_names = {"143.jpg", "000020.png", "original.jpg", "sample.jpg"}
    candidates = []
    for path in ROOT.rglob("*"):
        if not is_workspace_image(path):
            continue
        rank = 0 if path.name.lower() in preferred_names else 1
        candidates.append((rank, len(path.parts), str(path).lower(), path))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return [path for _, _, _, path in candidates]


def find_default_images(limit: int = 1) -> list[str]:
    candidates = iter_workspace_images()
    return [str(path) for path in candidates[:limit]]


def normalize_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def has_glob(pattern: str) -> bool:
    return any(char in pattern for char in GLOB_CHARS)


def resolve_image_pattern(pattern: str) -> list[Path]:
    raw = Path(pattern).expanduser()
    candidates = [raw] if raw.is_absolute() else [ROOT / raw, raw]
    matches: list[Path] = []

    for candidate in candidates:
        candidate_str = str(candidate)
        if has_glob(candidate_str):
            matches.extend(Path(p) for p in glob.glob(candidate_str, recursive=True))
            continue
        if candidate.is_dir():
            matches.extend(p for p in candidate.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        elif candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
            matches.append(candidate)
    return matches


def find_similar_images(pattern: str) -> list[Path]:
    target = Path(pattern)
    target_name = target.name.lower()
    target_stem = target.stem.lower()
    if not target_name:
        return []
    similar = []
    for image in iter_workspace_images():
        if image.name.lower() == target_name or image.stem.lower() == target_stem:
            similar.append(image)
    return similar


def unique_existing_images(paths: list[Path]) -> list[str]:
    unique: list[str] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(str(resolved))
    return unique


def collect_images(patterns: list[str] | None, limit: int = 1) -> tuple[list[str], list[str]]:
    if not patterns:
        return unique_existing_images([Path(p) for p in find_default_images(limit=limit)]), []

    resolved_paths: list[Path] = []
    notes: list[str] = []
    unmatched: list[tuple[str, list[Path]]] = []
    for pattern in patterns:
        matches = resolve_image_pattern(pattern)
        if matches:
            resolved_paths.extend(matches)
            continue

        similar = find_similar_images(pattern)
        if len(similar) == 1:
            resolved_paths.append(similar[0])
            notes.append(f"[auto-resolve] {pattern} -> {similar[0]}")
        else:
            unmatched.append((pattern, similar[:5]))

    images = unique_existing_images(resolved_paths)
    if images and not unmatched:
        return images, notes

    examples = iter_workspace_images()[:8]
    message = ["No images matched one or more --images inputs."]
    for pattern, similar in unmatched:
        message.append(f"  - {pattern}")
        if similar:
            message.append("    Similar files:")
            message.extend(f"      {path}" for path in similar)
    if examples:
        message.append("Available image examples:")
        message.extend(f"  {path}" for path in examples)
    raise FileNotFoundError("\n".join(message))


def split_deblur_tokens(values: list[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(token.strip().lower() for token in value.split(",") if token.strip())
    return tokens


def resolve_deblurs(names: list[str]) -> list[str]:
    tokens = split_deblur_tokens(names)
    if "all" in tokens:
        return ["wiener", "richardson_lucy", "mprnet", "nafnet", "lightweight_unet"]
    deblurs = []
    for name in tokens:
        if name not in DEBLUR_ALIASES:
            options = ", ".join(sorted([*DEBLUR_ALIASES.keys(), "all"]))
            raise ValueError(f"Unknown deblur preset: {name}. Available: {options}")
        deblur = DEBLUR_ALIASES[name]
        if deblur not in deblurs:
            deblurs.append(deblur)
    return deblurs or ["wiener"]


def make_run_tag(preset: str, deblurs: list[str], attack_mode: str = "per_deblur") -> str:
    names = [DEBLUR_DISPLAY_NAMES.get(deblur, deblur) for deblur in deblurs]
    if len(names) == 1:
        deblur_part = names[0]
    else:
        deblur_part = "multi-" + "-".join(names)
    if attack_mode == "ensemble":
        deblur_part = "ensemble-" + "-".join(names)
    return f"{preset}_{deblur_part}"


def first_existing_path(paths: list[str | Path | None]) -> Path | None:
    for path in paths:
        if not path:
            continue
        candidate = normalize_path(path)
        if candidate.exists():
            return candidate
    return None


def run_one(args: argparse.Namespace, deblur: str, images: list[str], base_output: Path) -> None:
    preset = PRESETS[args.preset]
    output_dir = base_output / deblur
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_deblur_amplified_attack.py"),
        "--config",
        str(ROOT / "configs" / "deblur_amplified.yaml"),
        "--deblur",
        deblur,
        "--output-dir",
        str(output_dir),
        "--image-size",
        str(args.image_size or preset["image_size"]),
        "--steps",
        str(args.steps if args.steps is not None else preset["steps"]),
        "--topk",
        str(args.topk or preset["topk"]),
        "--device",
        args.device,
        "--kernel-size",
        str(args.kernel_size or preset["kernel_size"]),
        "--mask",
        args.mask,
        "--images",
        *images,
    ]
    if args.epsilon is not None:
        cmd.extend(["--epsilon", str(args.epsilon)])
    if deblur == "lightweight_unet":
        checkpoint = first_existing_path(
            [
                args.cnn_checkpoint,
                ROOT / "output" / "deblur" / "lightweight_unet.pt",
            ]
        )
        if checkpoint is None:
            print(
                "[skip] lightweight_unet needs a checkpoint. Train one with:\n"
                "       python scripts/train_lightweight_deblur.py --images <sharp_images> --output output/deblur/lightweight_unet.pt",
                flush=True,
            )
            return
        cmd.extend(["--cnn-checkpoint", str(checkpoint)])
    if deblur == "mprnet":
        weights = first_existing_path(
            [
                args.mprnet_weights,
                ROOT / "weights" / "model_deblurring.pth",
                ROOT / "external" / "MPRNet" / "Deblurring" / "pretrained_models" / "model_deblurring.pth",
            ]
        )
        if weights is None:
            print(
                "[skip] mprnet needs official deblurring weights. Download with:\n"
                "       python scripts/download_deblur_weights.py --model mprnet\n"
                "or place model_deblurring.pth under weights/ or external/MPRNet/Deblurring/pretrained_models/",
                flush=True,
            )
            return
        cmd.extend(["--cnn-checkpoint", str(weights)])
    if deblur in {"nafnet", "nafnet32", "nafnet64"}:
        width = 64 if deblur == "nafnet64" else 32
        weights = first_existing_path(
            [
                args.nafnet64_weights if width == 64 else args.nafnet32_weights,
                args.nafnet_weights,
                ROOT / "weights" / f"NAFNet-GoPro-width{width}.pth",
                ROOT / "external" / "NAFNet" / "experiments" / "pretrained_models" / f"NAFNet-GoPro-width{width}.pth",
            ]
        )
        if weights is None:
            print(
                f"[skip] {deblur} needs official NAFNet GoPro weights. Download with:\n"
                f"       python scripts/download_deblur_weights.py --model nafnet{width}\n"
                "or place the .pth file under weights/ or external/NAFNet/experiments/pretrained_models/",
                flush=True,
            )
            return
        cmd.extend(["--cnn-checkpoint", str(weights)])

    print(f"\n=== Running {deblur} ===", flush=True)
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def run_ensemble(args: argparse.Namespace, deblurs: list[str], images: list[str], base_output: Path) -> None:
    preset = PRESETS[args.preset]
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_ensemble_deblur_amplified_attack.py"),
        "--config",
        str(ROOT / "configs" / "deblur_amplified.yaml"),
        "--output-dir",
        str(base_output),
        "--image-size",
        str(args.image_size or preset["image_size"]),
        "--steps",
        str(args.steps if args.steps is not None else preset["steps"]),
        "--topk",
        str(args.topk or preset["topk"]),
        "--device",
        args.device,
        "--kernel-size",
        str(args.kernel_size or preset["kernel_size"]),
        "--mask",
        args.mask,
        "--deblurs",
        *deblurs,
        "--images",
        *images,
    ]
    if args.epsilon is not None:
        cmd.extend(["--epsilon", str(args.epsilon)])
    if args.cnn_checkpoint:
        cmd.extend(["--cnn-checkpoint", args.cnn_checkpoint])
    if args.mprnet_weights:
        cmd.extend(["--mprnet-weights", args.mprnet_weights])
    if args.nafnet_weights:
        cmd.extend(["--nafnet-weights", args.nafnet_weights])
    if args.nafnet32_weights:
        cmd.extend(["--nafnet32-weights", args.nafnet32_weights])
    if args.nafnet64_weights:
        cmd.extend(["--nafnet64-weights", args.nafnet64_weights])

    print("\n=== Running shared ensemble attack ===", flush=True)
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convenient runner for deblur-amplified attack experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=PRESETS.keys(), default="quick")
    parser.add_argument(
        "--deblur",
        nargs="+",
        default=["wiener"],
        help="One or more deblur methods. Examples: --deblur wiener rl mprnet, --deblur all, or --deblur wiener,rl,mprnet.",
    )
    parser.add_argument(
        "--attack-mode",
        choices=["per_deblur", "ensemble"],
        default="per_deblur",
        help="per_deblur = optimize one perturbation per deblur method; ensemble = optimize one shared perturbation against all selected methods.",
    )
    parser.add_argument("--images", nargs="*", help="Image files, folders, or glob patterns.")
    parser.add_argument("--output", default=None, help="Output root. Defaults to output/easy_runs/<timestamp>_<preset>_<deblur-tag>.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cnn-checkpoint", default=None)
    parser.add_argument("--mprnet-weights", default=None)
    parser.add_argument("--nafnet-weights", default=None)
    parser.add_argument("--nafnet32-weights", default=None)
    parser.add_argument("--nafnet64-weights", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--mask", choices=["on", "off"], default="on", help="on = local detector-guided mask, off = global perturbation.")
    parser.add_argument("--limit", type=int, default=1, help="Number of auto-selected images when --images is omitted.")
    args = parser.parse_args()

    deblurs = resolve_deblurs(args.deblur)
    images, image_notes = collect_images(args.images, limit=args.limit)
    if not images:
        raise FileNotFoundError("No images provided and no default sample images found in this workspace.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = make_run_tag(args.preset, deblurs, attack_mode=args.attack_mode)
    base_output = normalize_path(args.output) if args.output else ROOT / "output" / "easy_runs" / f"{stamp}_{run_tag}"
    base_output.mkdir(parents=True, exist_ok=True)

    run_record = {
        "timestamp": stamp,
        "entry_command": " ".join(sys.argv),
        "argv": sys.argv,
        "cwd": str(ROOT),
        "python": sys.executable,
        "preset": args.preset,
        "preset_values": PRESETS[args.preset],
        "attack_mode": args.attack_mode,
        "requested_deblur": args.deblur,
        "resolved_deblurs": deblurs,
        "images": images,
        "image_resolution_notes": image_notes,
        "output": str(base_output),
        "args": vars(args),
    }
    with open(base_output / "run_args.json", "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2, ensure_ascii=False)

    print("Images:", flush=True)
    for note in image_notes:
        print("  ", note, flush=True)
    for image in images:
        print("  ", image, flush=True)
    print("Deblur methods:", ", ".join(deblurs), flush=True)
    print("Output:", base_output, flush=True)

    if args.attack_mode == "ensemble":
        run_ensemble(args, deblurs, images, base_output)
    else:
        for deblur in deblurs:
            run_one(args, deblur, images, base_output)

    print("\nDone. Check panels and metrics under:", flush=True)
    print(base_output, flush=True)


if __name__ == "__main__":
    main()
