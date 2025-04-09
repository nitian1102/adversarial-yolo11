# core/models/adversarial_yolo.py
from ultralytics import YOLO
import torch
from core.attacks import get_attack


class AdversarialYOLO(YOLO):
    """支持对抗攻击的YOLO模型"""

    def __init__(self, model_path: str, attack_method: str = "fgsm", **attack_params):
        super().__init__(model_path)
        self.attack_method = attack_method.lower()
        self.attack = get_attack(attack_method)(**attack_params)  # 初始化攻击实例

    def predict_attack(self, image: torch.Tensor) -> torch.Tensor:
        """执行对抗攻击并返回扰动后的图像"""
        return self.attack.execute(self.model, image.to(self.device))