# core/attacks/fgsm.py
import torch
from .base_attack import BaseAttack


class FGSMAttack(BaseAttack):
    """增强版FGSM攻击"""

    def required_params(self):
        return ['epsilon']

    def execute(self, model, image: torch.Tensor) -> torch.Tensor:
        image.requires_grad = True
        pred = self.get_predictions(model, image)
        loss = self.calculate_loss(pred)
        model.zero_grad()
        loss.backward()

        # 生成扰动
        grad = image.grad.data
        perturbation = self.params['epsilon'] * grad.sign()

        # 应用扰动并裁剪
        perturbed = image + perturbation
        return torch.clamp(perturbed, 0, 1).detach()