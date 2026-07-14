# Deblur-Amplified Attack Code

This package implements the current research direction:

1. Classical deblur baselines: Wiener filtering and Richardson-Lucy deconvolution.
2. Lightweight CNN deblur baseline: a compact residual UNet trained on synthetic motion blur pairs.
3. Core attack: detector-guided, region-aware, deblur-amplified perturbation for YOLOv5s.

The attack optimizes detector degradation after deblurring, while penalizing
pre-deblur detector degradation so the residual remains comparatively dormant
before restoration.

## Run The Main Attack

The easiest entry point is:

```powershell
python run_experiment.py
```

This uses the `quick` preset, automatically selects one sample image from the
workspace, runs the Wiener baseline, and writes results to `output/easy_runs/`.

Common presets:

```powershell
python run_experiment.py --preset quick
python run_experiment.py --preset dev --deblur rl
python run_experiment.py --preset dev --deblur mprnet
python run_experiment.py --preset dev --deblur wiener rl mprnet
python run_experiment.py --preset dev --deblur wiener,rl,mprnet
python run_experiment.py --preset formal --deblur wiener --images "path/to/images/*.jpg"
python run_experiment.py --preset dev --deblur all --images "path/to/images/*.jpg"
```

`--deblur` accepts multiple methods in one run. The motion blur settings are
shared. By default, each deblur method gets its own optimized perturbation and
subfolder under the same run root, for example:

```text
output/easy_runs/20260709_153000_dev_multi-wiener-rl-mprnet/
  wiener/
  richardson_lucy/
  mprnet/
```

By default, run folder names include the preset and deblur tag. If `--output`
is provided, that explicit folder is used instead.

Shared ensemble attack:

```powershell
python run_experiment.py --preset dev --attack-mode ensemble --deblur wiener rl mprnet --images "images/原图.JPG"
python run_experiment.py --preset dev --attack-mode ensemble --deblur wiener rl mprnet nafnet --images "images/原图.JPG"
```

`--attack-mode ensemble` optimizes one shared perturbation against all selected
deblur methods. Outputs are saved as:

```text
output/easy_runs/20260709_153000_dev_ensemble-wiener-rl-mprnet/
  0000_原图/
    B0_clean.png
    B1_blur.png
    B4_adv_blur.png
    delta_abs.png
    region_mask.png
    wiener/
    richardson_lucy/
    mprnet/
```

Use this mode for transferability/universal perturbation experiments. Use the
default mode for white-box per-deblur upper-bound experiments.

Mask control:

```powershell
python run_experiment.py --preset dev --mask on
python run_experiment.py --preset dev --mask off
```

- `--mask on`: detector-guided local perturbation, the default.
- `--mask off`: global perturbation over the whole blurred image.

Preset meanings:

- `quick`: 160 px, 1 step, 9 px blur kernel, for smoke tests.
- `dev`: 320 px, 5 steps, 15 px blur kernel, for parameter checks.
- `formal`: 640 px, 20 steps, 21 px blur kernel, for real experiments.

Windows shortcut:

```powershell
.\run_quick.ps1
.\run_quick.ps1 -Deblur rl
.\run_quick.ps1 -Deblur wiener,rl,mprnet
```

The lower-level script is still available when you need full control:

```powershell
python scripts/run_deblur_amplified_attack.py --config configs/deblur_amplified.yaml
```

Use Richardson-Lucy:

```powershell
python scripts/run_deblur_amplified_attack.py --deblur richardson_lucy --images "path/to/images/*.jpg"
```

Use the lightweight CNN:

```powershell
python scripts/run_deblur_amplified_attack.py --deblur lightweight_unet --cnn-checkpoint output/deblur/lightweight_unet.pt --images "path/to/images/*.jpg"
```

Use the official MPRNet deblurring model:

```powershell
python scripts/download_deblur_weights.py --model mprnet
python run_experiment.py --preset dev --deblur mprnet --images "images/原图.JPG"
```

MPRNet is stronger than the classical Wiener/Richardson-Lucy baselines. Use it
for the main learned-deblur experiments; keep classical methods as explanatory
baselines.

Use official NAFNet GoPro weights:

```powershell
python scripts/download_deblur_weights.py --model nafnet32
python run_experiment.py --preset dev --deblur nafnet --images "path/to/images/*.jpg"
```

If Google Drive download is slow, manually download and place weights here:

- MPRNet: `weights/model_deblurring.pth` or `external/MPRNet/Deblurring/pretrained_models/model_deblurring.pth`
- NAFNet width32: `weights/NAFNet-GoPro-width32.pth` or `external/NAFNet/experiments/pretrained_models/NAFNet-GoPro-width32.pth`
- NAFNet width64: `weights/NAFNet-GoPro-width64.pth` or `external/NAFNet/experiments/pretrained_models/NAFNet-GoPro-width64.pth`

Official repositories:

- MPRNet: https://github.com/swz30/MPRNet
- NAFNet: https://github.com/megvii-research/NAFNet
- Restormer: https://github.com/swz30/Restormer

The script writes each sample under `output/deblur_amplified/`:

- `run_args.json` at the run root when using `run_experiment.py`
- `resolved_config.json` under each deblur output folder
- `summary.json` under each deblur output folder
- `B0_clean.png`
- `B1_blur.png`
- `B2_deblur.png`
- `B4_adv_blur.png`
- `B5_adv_deblur.png`
- `delta_abs.png`
- `region_mask.png`
- `panel.png`
- `metrics.json`

## Train The Lightweight CNN Deblur Baseline

```powershell
python scripts/train_lightweight_deblur.py --images "path/to/sharp/images" --output output/deblur/lightweight_unet.pt
```

This trains on synthetic motion blur pairs generated online. It is intended as
a controllable lightweight CNN baseline, not as a state-of-the-art restoration model.

## Important Metrics

`metrics.json` includes:

- `pre_gap`: detector confidence drop before deblurring.
- `post_gap`: detector confidence drop after deblurring.
- `cascade_amplification_ratio`: `post_gap / (pre_gap + eps)`.
- `deblur_l1_change`: how much the restored image changed after the residual.

A successful deblur-revealed effect should have small `pre_gap`, larger
`post_gap`, and high `cascade_amplification_ratio`.
