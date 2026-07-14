# main.py
import torch
from core.models import AdversarialYOLO
from core.utils import image_utils, visualization, config
from PIL import Image
import glob
import os
import numpy as np

# DATA_PATHS = []
# target_dir = r"E:\git_project\darknet-linux\darknet\test\images"
# new_paths = glob.glob(os.path.join(target_dir, "*.jpg"))
# DATA_PATHS.extend(new_paths)


DATA_PATHS = [
    "F:/Dataset/KITTI/csy_lidar/image_2/000020.png",
'F:/Dataset/KITTI/csy_lidar/image_2/000020.png',
'F:/Dataset/KITTI/csy_lidar/image_2/000090.png',
'F:/Dataset/KITTI/csy_lidar/image_2/000263.png',
'F:/Dataset/KITTI/csy_lidar/image_2/000381.png',
'F:/Dataset/KITTI/csy_lidar/image_2/000414.png',
'F:/Dataset/KITTI/csy_lidar/image_2/000546.png',
'F:/Dataset/KITTI/csy_lidar/image_2/000746.png',
'F:/Dataset/KITTI/csy_lidar/image_2/000779.png',
'F:/Dataset/KITTI/csy_lidar/image_2/000784.png',
'F:/Dataset/KITTI/csy_lidar/image_2/000815.png',
'F:/Dataset/KITTI/csy_lidar/image_2/001018.png',
'F:/Dataset/KITTI/csy_lidar/image_2/001066.png',
'F:/Dataset/KITTI/csy_lidar/image_2/001067.png',
'F:/Dataset/KITTI/csy_lidar/image_2/001239.png',
'F:/Dataset/KITTI/csy_lidar/image_2/001341.png',
'F:/Dataset/KITTI/csy_lidar/image_2/001365.png',
'F:/Dataset/KITTI/csy_lidar/image_2/001519.png',
'F:/Dataset/KITTI/csy_lidar/image_2/001758.png',
'F:/Dataset/KITTI/csy_lidar/image_2/001984.png',
'F:/Dataset/KITTI/csy_lidar/image_2/002117.png',
]


def main():
    # 加载攻击配置
    cfg = config.load_config('configs/pgd.yaml')

    # 初始化对抗模型
    model = AdversarialYOLO(
        model_path=cfg['paths']['model_path'],
        attack_method=cfg['attack']['method'],
        **cfg['attack']['params']
    )

    # 加载和处理图像
    originals = [Image.open(img_path) for img_path in DATA_PATHS]
    attacked_images = [
        model.predict_attack(image_utils.process_image(img, device=model.device)) for img in originals
        # img for img in originals
    ]
    #将所有攻击后的图片保存
    # attacked_save_path=[]
    # for i,adv in enumerate(attacked_images):
    #     original_size = originals[i].size
    #     adv_img_np = adv.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255
    #     adv_img_np = adv_img_np.astype(np.uint8)
    #     # 创建PIL图像对象
    #     adv_img_pil = Image.fromarray(adv_img_np)
    #
    #     # 使用高质量插值方法恢复原始尺寸
    #     adv_img_resized = adv_img_pil.resize(original_size, Image.BICUBIC)
    #
    #     attack_dir = os.path.join(cfg['paths']['save_dir'], "attacked_images")
    #     os.makedirs(attack_dir, exist_ok=True)
    #     attacked_image_path = os.path.join(attack_dir, f"attacked_{i}.png")
    #     adv_img_resized.save(attacked_image_path)
    #     attacked_save_path.append(attacked_image_path)
    # print(f'被攻击图片地址为,{attacked_save_path}')
    # attacked_save_img = [Image.open(img_path) for img_path in attacked_save_path]

    # 执行攻击
    # attacked_tensors = [model.predict_attack(img) for img in processed]

    # 结果可视化
    # adv_img = image_utils.tensor_to_image(adv_tensor)
    # visualization.visualize_attack_comparison(
    #         originals,
    #         attacked_save_img,
    #         model_path=cfg['paths']['model_path'],
    #         save_dir=cfg['paths']['save_dir'],
    #         confidence_threshold=cfg['attack'].get('confidence_threshold')
    #     )

    visualization.visualize_attack_comparison(
            originals,
            attacked_images,
            model_path=cfg['paths']['model_path'],
            save_dir=cfg['paths']['save_dir'],
            confidence_threshold=cfg['attack'].get('confidence_threshold')
        )


if __name__ == "__main__":
    main()