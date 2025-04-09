# core/attacks/pgd.py
import torch
from .base_attack import BaseAttack


class PGDAttack(BaseAttack):
    """PGD攻击实现"""

    def required_params(self):
        return ['epsilon', 'step_size', 'num_steps']

    def execute(self, model, image: torch.Tensor) -> torch.Tensor:
        epsilon = float(self.params['epsilon'])
        step_size = float(self.params['step_size'])
        num_steps = int(self.params['num_steps'])

        original = image.clone().detach()

        perturbed = image + torch.zeros_like(image).uniform_(-epsilon, epsilon)
        perturbed = torch.clamp(perturbed, 0, 1).detach()

        for _ in range(num_steps):
            perturbed.requires_grad = True
            pred = self.get_predictions(model, perturbed)
            loss = self.calculate_loss(pred)
            model.zero_grad()
            if perturbed.grad is not None:
                perturbed.grad.zero_()
            loss.backward()

            with torch.no_grad():
                grad = perturbed.grad.sign()
                perturbed = perturbed + step_size * grad
                perturbed = torch.max(torch.min(perturbed, original + epsilon),original - epsilon)
                perturbed = torch.clamp(perturbed, 0, 1).detach()

        return perturbed