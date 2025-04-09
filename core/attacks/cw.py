import torch
import torch.optim as optim
from .base_attack import BaseAttack


class CWAttack(BaseAttack):
    """Carlini & Wagner (CW) L2攻击实现"""

    def required_params(self):
        return ['c', 'lr', 'num_steps']

    def execute(self, model, image: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(image, requires_grad=True)
        optimizer = optim.Adam([delta], lr=self.params['lr'])

        for _ in range(int(self.params['num_steps'])):
            perturbed = torch.clamp(image + delta, 0, 1)
            predictions = self.get_predictions(model, perturbed)

            # 计算损失
            objectness = predictions[:, 4]
            loss_confidence = objectness.sum()
            loss_perturbation = torch.norm(delta, p=2)
            total_loss = loss_confidence + self.params['c'] * loss_perturbation

            # 优化步骤
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 保持扰动范围
            delta.data = torch.clamp(
                delta,
                -self.params.get('epsilon', 0.1),
                self.params.get('epsilon', 0.1)
            )

        return torch.clamp(image + delta, 0, 1).detach()