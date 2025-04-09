# core/utils/visualization.py
import os
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
from typing import List
from core.utils import image_utils
import numpy as np


def visualize_attack_comparison(
        original_images: List[Image.Image],
        attacked_images: List[torch.Tensor],
        model_path: str,
        save_dir: str = "output",
        confidence_threshold: float = 0.5

):
    """Visualize and save results of adversarial attacks."""
    os.makedirs(save_dir, exist_ok=True)
    attack_dir = os.path.join(save_dir, "attacked_images")
    os.makedirs(attack_dir, exist_ok=True)

    model = YOLO(model_path)

    for i, (orig, adv) in enumerate(zip(original_images, attacked_images)):
        orig_resized = image_utils.resize_with_padding(orig, target_size=(640, 640))
        result_orig = model.predict(orig_resized, imgsz=(640, 640), conf=confidence_threshold)[0]
        result_adv = model.predict(adv, imgsz=(640, 640), conf=confidence_threshold)[0]

        img_orig = result_orig.plot(line_width=2)
        img_adv = result_adv.plot(line_width=2)

        adv_img_np = adv.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255
        adv_img_np = adv_img_np.astype(np.uint8)

        attacked_image_path = os.path.join(attack_dir, f"attacked_{i}.png")
        Image.fromarray(adv_img_np).save(attacked_image_path)

        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        axs[0].imshow(orig_resized)
        axs[0].set_title("Original Image", fontsize=12)
        axs[0].axis("off")

        axs[1].imshow(img_orig[..., ::-1])
        axs[1].set_title("Original Image with Detection", fontsize=12)
        axs[1].axis("off")

        axs[2].imshow(img_adv)
        axs[2].set_title("Attacked Image with Detection", fontsize=12)
        axs[2].axis("off")

        visualization_path = os.path.join(save_dir, f"visualization_{i+1}.png")
        plt.tight_layout()
        plt.savefig(visualization_path)
        plt.close()

        print(f"Visualization saved to {visualization_path}")
        print(f"Attacked image saved to {attacked_image_path}")