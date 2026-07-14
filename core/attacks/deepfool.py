# core/attacks/deepfool.py
import torch
from .base_attack import BaseAttack


class DeepFoolAttack(BaseAttack):
    """Optimized DeepFool attack implementation."""

    def required_params(self):
        return ['num_steps', 'epsilon']

    def execute(self, model, image: torch.Tensor) -> torch.Tensor:
        """
        Perform the DeepFool attack to generate adversarial examples.

        :param model: The target model.
        :param image: The input image tensor.
        :return: The adversarially perturbed image tensor.
        """
        num_steps = int(self.params['num_steps'])
        epsilon = float(self.params.get('epsilon', 1e-6))  # Default epsilon if not provided
        perturbed = image.clone().detach()
        perturbed.requires_grad = True

        for _ in range(num_steps):
            # Get predictions and gradients
            pred = self.get_predictions(model, perturbed)  # Shape: [batch_size, num_classes]
            current_class = pred.argmax(dim=1)  # Current predicted class

            model.zero_grad()
            if perturbed.grad is not None:
                perturbed.grad.zero_()

            # Compute loss for the current class
            loss = pred[0, current_class].sum()  # Focus on the current class logits
            loss.backward(retain_graph=True)

            # Compute gradient and normalize it
            grad = perturbed.grad.data
            with torch.no_grad():
                perturbation = grad / (torch.norm(grad, p=2) + 1e-8)  # Normalize gradient
                perturbed = perturbed + epsilon * perturbation  # Apply perturbation
                perturbed = torch.clamp(perturbed, 0, 1).detach()  # Ensure valid pixel range
                perturbed.requires_grad = True

        return perturbed

    def _check_params(self):
        """
        Validate the parameters for the DeepFool attack.
        """
        if 'num_steps' not in self.params or not isinstance(self.params['num_steps'], int) or self.params['num_steps'] <= 0:
            raise ValueError("The parameter 'num_steps' must be a positive integer.")
        if 'epsilon' in self.params and (not isinstance(self.params['epsilon'], float) or self.params['epsilon'] < 0):
            raise ValueError("The parameter 'epsilon' must be a non-negative float.")