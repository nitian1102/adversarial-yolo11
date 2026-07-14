# core/attacks/zoo.py
import torch
from .base_attack import BaseAttack


class ZOOAttack(BaseAttack):
    """Zeroth Order Optimization (ZOO)攻击实现"""

    def required_params(self):
        return ['epsilon', 'lr', 'num_steps']

    def execute(self, model, image: torch.Tensor) -> torch.Tensor:
        epsilon = float(self.params['epsilon'])
        lr = float(self.params['lr'])
        num_steps = int(self.params['num_steps'])

        delta = torch.zeros_like(image, requires_grad=True)
        perturbed = image.clone().detach()

        for _ in range(num_steps):
            perturbed = torch.clamp(image + delta, 0, 1)
            loss = self.calculate_loss(self.get_predictions(model, perturbed))

            # Zeroth-order gradient estimation
            grad_estimate = torch.zeros_like(delta)
            for i in range(delta.numel()):
                delta_flat = delta.reshape(-1)
                delta_flat[i] += epsilon
                perturbed_plus = torch.clamp(image + delta.view_as(image), 0, 1)
                loss_plus = self.calculate_loss(self.get_predictions(model, perturbed_plus))

                delta_flat[i] -= 2 * epsilon
                perturbed_minus = torch.clamp(image + delta.view_as(image), 0, 1)
                loss_minus = self.calculate_loss(self.get_predictions(model, perturbed_minus))

                grad_estimate.reshape(-1)[i] = (loss_plus - loss_minus) / (2 * epsilon)
                delta_flat[i] += epsilon

            # Update delta
            delta = delta - lr * grad_estimate
            delta = torch.clamp(delta, -epsilon, epsilon)

        return torch.clamp(image + delta, 0, 1).detach()