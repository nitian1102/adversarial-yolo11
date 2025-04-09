# core/utils/image_utils.py
from PIL import Image
import torch
import numpy as np
from typing import  Tuple


def load_images(paths: list) -> list:
    """加载图像列表"""
    return [Image.open(p) for p in paths]

def process_image(
    img: Image.Image,
    target_size: tuple = (640, 640),
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """统一图像预处理"""
    img_resized = img.resize(target_size)
    tensor = torch.tensor(
        np.array(img_resized).transpose(2,0,1)/255.0,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)
    return tensor

def resize_with_padding(img: Image.Image, target_size: Tuple[int, int] = (640, 640)) -> Image.Image:
    """Resize the image to fit within the target size, adding black padding if necessary."""
    original_width, original_height = img.size
    target_width, target_height = target_size

    aspect_ratio = original_width / original_height

    if aspect_ratio > 1:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    img_resized = img.resize((new_width, new_height))
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    new_img.paste(img_resized, ((target_width - new_width) // 2, (target_height - new_height) // 2))

    return new_img

# def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
#     """Tensor转PIL图像"""
#     arr = tensor.squeeze(0).cpu().numpy().transpose(1,2,0)*255
#     return Image.fromarray(arr.astype(np.uint8))