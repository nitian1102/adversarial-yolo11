import argparse
from pathlib import Path
import re
import subprocess
import sys

import requests
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "mprnet": {
        "repo": "https://github.com/swz30/MPRNet.git",
        "repo_dir": ROOT / "external" / "MPRNet",
        "file_id": "1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb",
        "output": ROOT / "external" / "MPRNet" / "Deblurring" / "pretrained_models" / "model_deblurring.pth",
        "source": "Official MPRNet Deblurring model",
    },
    "nafnet32": {
        "repo": "https://github.com/megvii-research/NAFNet.git",
        "repo_dir": ROOT / "external" / "NAFNet",
        "file_id": "1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj",
        "output": ROOT / "external" / "NAFNet" / "experiments" / "pretrained_models" / "NAFNet-GoPro-width32.pth",
        "source": "Official NAFNet GoPro width32 model",
    },
    "nafnet64": {
        "repo": "https://github.com/megvii-research/NAFNet.git",
        "repo_dir": ROOT / "external" / "NAFNet",
        "file_id": "1S0PVRbyTakYY9a82kujgZLbMihfNBLfC",
        "output": ROOT / "external" / "NAFNet" / "experiments" / "pretrained_models" / "NAFNet-GoPro-width64.pth",
        "source": "Official NAFNet GoPro width64 model",
    },
}


def ensure_repo(model: str) -> None:
    info = MODELS[model]
    repo_dir = info["repo_dir"]
    if repo_dir.exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", info["repo"], str(repo_dir)], cwd=str(ROOT), check=True)


def _get_confirm_token(response: requests.Response) -> str | None:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    match = re.search(r"confirm=([0-9A-Za-z_]+)", response.text)
    if match:
        return match.group(1)
    return None


def download_google_drive(file_id: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    url = "https://docs.google.com/uc?export=download"
    response = session.get(url, params={"id": file_id}, stream=True)
    token = _get_confirm_token(response)
    if token:
        response.close()
        response = session.get(url, params={"id": file_id, "confirm": token}, stream=True)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type.lower():
        raise RuntimeError(
            "Google Drive returned an HTML page instead of a weight file. "
            "Open the official link in a browser and manually place the file at: "
            f"{output}"
        )

    total = int(response.headers.get("content-length", 0))
    tmp = output.with_suffix(output.suffix + ".part")
    with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=output.name) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            bar.update(len(chunk))
    tmp.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official deblur model weights used by this project.")
    parser.add_argument("--model", choices=MODELS.keys(), default="mprnet")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    info = MODELS[args.model]
    ensure_repo(args.model)
    output = info["output"]
    if output.exists() and not args.force:
        print(f"Already exists: {output}")
        return
    print(f"Downloading {info['source']} to {output}")
    download_google_drive(info["file_id"], output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise
