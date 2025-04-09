# main.py
import torch
from core.models import AdversarialYOLO
from core.utils import image_utils, visualization, config
from PIL import Image


DATA_PATHS = [
    "E:/Tsinghua Data/tt100k_2021-yolo/tt100k_yolo/images/143.jpg",
    "E:/Tsinghua Data/tt100k_2021-yolo/tt100k_yolo/images/148.jpg",
    "E:/Tsinghua Data/tt100k_2021-yolo/tt100k_yolo/images/295.jpg",
    "E:/Tsinghua Data/tt100k_2021-yolo/tt100k_yolo/images/1719.jpg",
]


def main():
    # 加载攻击配置
    cfg = config.load_config('configs/cw.yaml')

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
    ]
    # 执行攻击
    # attacked_tensors = [model.predict_attack(img) for img in processed]

    # 结果可视化
    # adv_img = image_utils.tensor_to_image(adv_tensor)
    visualization.visualize_attack_comparison(
            originals,
            attacked_images,
            model_path=cfg['paths']['model_path'],
            save_dir=cfg['paths']['save_dir'],
            confidence_threshold=cfg['attack'].get('confidence_threshold')
        )


if __name__ == "__main__":
    main()