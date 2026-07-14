import argparse
from pathlib import Path
import random
import sys

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.deblurs.lightweight_cnn import SimpleDeblurUNet
from core.losses import edge_difference
from core.transforms import apply_motion_blur, build_motion_kernel


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolderDataset(Dataset):
    def __init__(self, root: str, image_size: int = 256):
        self.root = Path(root)
        self.paths = [p for p in self.root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        img = TF.resize(img, [self.image_size, self.image_size])
        if random.random() < 0.5:
            img = TF.hflip(img)
        return TF.to_tensor(img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the lightweight UNet deblur baseline on synthetic blur pairs.")
    parser.add_argument("--images", required=True, help="Folder containing sharp training images.")
    parser.add_argument("--output", default="output/deblur/lightweight_unet.pt")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--kernel-min", type=int, default=9)
    parser.add_argument("--kernel-max", type=int, default=25)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    dataset = ImageFolderDataset(args.images, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    model = SimpleDeblurUNet(width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for sharp in loader:
            sharp = sharp.to(device)
            kernel_size = random.randrange(args.kernel_min, args.kernel_max + 1, 2)
            angle = random.uniform(-35.0, 35.0)
            kernel = build_motion_kernel(kernel_size, angle, channels=3, device=device)
            blurred = apply_motion_blur(sharp, kernel)
            restored = model(blurred)
            loss_l1 = (restored - sharp).abs().mean()
            loss_edge = edge_difference(restored, sharp)
            loss = loss_l1 + 0.1 * loss_edge
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * sharp.shape[0]
        avg = total / len(dataset)
        print(f"epoch={epoch + 1:03d} loss={avg:.6f}")
        torch.save(
            {"model": model.state_dict(), "epoch": epoch + 1, "loss": avg, "width": args.width},
            output,
        )
    print(f"saved checkpoint: {output}")


if __name__ == "__main__":
    main()
