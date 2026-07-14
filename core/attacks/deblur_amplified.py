from dataclasses import dataclass
from typing import Mapping

import torch

from core.detectors import YOLOv5Detector
from core.losses import edge_difference, edge_map, normalize_map, total_variation


@dataclass
class DeblurAmplifiedAttackConfig:
    epsilon: float = 8 / 255
    step_size: float = 2 / 255
    steps: int = 20
    topk: int = 40
    dormancy_tolerance: float = 0.03
    lambda_pre: float = 1.0
    lambda_vis: float = 0.15
    lambda_bg: float = 0.25
    lambda_tv: float = 0.03
    beta_deblur: float = 0.35
    beta_edge: float = 0.15
    beta_amp: float = 0.05
    mask_threshold: float = 0.55
    mask_blur: int = 17
    use_region_mask: bool = True
    saliency_weight: float = 0.65
    box_weight: float = 0.25
    edge_weight: float = 0.10
    init_mode: str = "amplification"


class RegionAwareDeblurAmplifiedAttack:
    """Detector-guided deblur-amplified perturbation.

    The attack optimizes detector degradation after deblurring, while requiring
    the same residual to remain comparatively dormant before deblurring.
    """

    def __init__(
        self,
        detector: YOLOv5Detector,
        deblur_model: torch.nn.Module,
        config: DeblurAmplifiedAttackConfig | None = None,
    ):
        self.detector = detector
        self.deblur_model = deblur_model
        self.config = config or DeblurAmplifiedAttackConfig()

    def _smooth_mask(self, mask: torch.Tensor) -> torch.Tensor:
        k = int(self.config.mask_blur)
        if k <= 1:
            return mask
        if k % 2 == 0:
            k += 1
        pad = k // 2
        weight = torch.ones((1, 1, k, k), device=mask.device, dtype=mask.dtype) / float(k * k)
        mask = torch.nn.functional.conv2d(
            torch.nn.functional.pad(mask, (pad, pad, pad, pad), mode="reflect"),
            weight,
        )
        return normalize_map(mask)

    def build_region_mask(self, blurred: torch.Tensor, deblurred: torch.Tensor) -> torch.Tensor:
        probe = deblurred.detach().clone().requires_grad_(True)
        pred = self.detector.raw(probe)
        ref_indices = self.detector.topk_indices(pred.detach(), topk=self.config.topk)
        confidence = self.detector.gathered_confidence(pred, ref_indices).mean()
        grad = torch.autograd.grad(confidence, probe, retain_graph=False, create_graph=False)[0]
        saliency = normalize_map(grad.abs().mean(dim=1, keepdim=True))

        box_mask = self.detector.prediction_box_mask(
            deblurred.detach(),
            pred=pred.detach(),
            topk=max(1, min(10, self.config.topk)),
        )
        edge = normalize_map(edge_map(deblurred.detach()))
        mask = (
            self.config.saliency_weight * saliency
            + self.config.box_weight * box_mask
            + self.config.edge_weight * edge
        )
        mask = normalize_map(mask)
        mask = self._smooth_mask(mask)
        hard = (mask >= self.config.mask_threshold).to(mask.dtype)
        # Keep a soft floor so gradients are not completely blocked.
        return (0.15 * mask + 0.85 * hard).clamp(0.0, 1.0).detach()

    def _frequency_guided(self, grad: torch.Tensor) -> torch.Tensor:
        h, w = grad.shape[-2:]
        hint = None
        if hasattr(self.deblur_model, "amplification_hint"):
            hint = self.deblur_model.amplification_hint(h, w, grad.device)
        if hint is None:
            return grad
        hint = hint.to(device=grad.device, dtype=grad.dtype)
        while hint.ndim < grad.ndim:
            hint = hint.unsqueeze(0)
        fft = torch.fft.fft2(grad)
        guided = torch.fft.ifft2(fft * (1.0 + hint)).real
        return guided

    @staticmethod
    def _remove_projection(source: torch.Tensor, basis: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        src = source.flatten(start_dim=1)
        bas = basis.flatten(start_dim=1)
        coeff = (src * bas).sum(dim=1, keepdim=True) / bas.square().sum(dim=1, keepdim=True).clamp_min(eps)
        out = src - coeff * bas
        return out.view_as(source)

    def _initial_delta(
        self,
        blurred: torch.Tensor,
        ref_post_indices: torch.Tensor,
        ref_pre_indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.init_mode == "zero":
            return torch.zeros_like(blurred)
        if self.config.init_mode == "random":
            return torch.empty_like(blurred).uniform_(-self.config.epsilon, self.config.epsilon)

        probe = blurred.detach().clone().requires_grad_(True)
        post = self.deblur_model(probe)
        post_obj = self.detector.hide_objective(self.detector.raw(post), ref_post_indices)
        g_post = torch.autograd.grad(post_obj, probe, retain_graph=False, create_graph=False)[0]

        probe_pre = blurred.detach().clone().requires_grad_(True)
        pre_obj = self.detector.hide_objective(self.detector.raw(probe_pre), ref_pre_indices)
        g_pre = torch.autograd.grad(pre_obj, probe_pre, retain_graph=False, create_graph=False)[0]

        guided = self._remove_projection(g_post, g_pre)
        guided = self._frequency_guided(guided)
        return self.config.epsilon * torch.sign(guided * (0.2 + 0.8 * mask))

    def run(self, blurred: torch.Tensor) -> dict[str, torch.Tensor | dict[str, float]]:
        cfg = self.config
        blurred = blurred.detach().to(self.detector.device).clamp(0.0, 1.0)
        with torch.no_grad():
            deblurred = self.deblur_model(blurred).detach()
            pred_post_ref = self.detector.raw(deblurred)
            pred_pre_ref = self.detector.raw(blurred)
            ref_post_indices = self.detector.topk_indices(pred_post_ref, topk=cfg.topk)
            ref_pre_indices = self.detector.topk_indices(pred_pre_ref, topk=cfg.topk)

        if cfg.use_region_mask:
            mask = self.build_region_mask(blurred, deblurred)
        else:
            mask = torch.ones(
                (blurred.shape[0], 1, blurred.shape[2], blurred.shape[3]),
                device=blurred.device,
                dtype=blurred.dtype,
            )
        mask3 = mask.repeat(1, blurred.shape[1], 1, 1)
        delta = self._initial_delta(blurred, ref_post_indices, ref_pre_indices, mask).detach()
        delta = delta.clamp(-cfg.epsilon, cfg.epsilon)

        history: list[float] = []
        for _ in range(cfg.steps):
            delta = delta.detach().requires_grad_(True)
            effective_delta = mask3 * delta
            adv_blur = (blurred + effective_delta).clamp(0.0, 1.0)
            adv_deblur = self.deblur_model(adv_blur)

            pred_post = self.detector.raw(adv_deblur)
            pred_pre = self.detector.raw(adv_blur)
            attack_obj = self.detector.hide_objective(pred_post, ref_post_indices)
            pre_penalty = self.detector.confidence_drop_penalty(
                pred_pre,
                pred_pre_ref,
                ref_pre_indices,
                tolerance=cfg.dormancy_tolerance,
            )

            masked_change = (mask * (adv_deblur - deblurred).abs()).sum() / mask.sum().clamp_min(1.0)
            edge_change = edge_difference(adv_deblur, deblurred, mask=mask)
            input_change = (mask * (adv_blur - blurred).abs()).sum() / mask.sum().clamp_min(1.0)
            amplification = masked_change / input_change.clamp_min(1e-6)
            visual = effective_delta.abs().mean()
            background = ((1.0 - mask3) * delta.abs()).mean()
            tv = total_variation(effective_delta)

            objective = (
                attack_obj
                + cfg.beta_deblur * masked_change
                + cfg.beta_edge * edge_change
                + cfg.beta_amp * amplification
                - cfg.lambda_pre * pre_penalty
                - cfg.lambda_vis * visual
                - cfg.lambda_bg * background
                - cfg.lambda_tv * tv
            )
            grad = torch.autograd.grad(objective, delta, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                delta = delta + cfg.step_size * torch.sign(grad)
                delta = delta.clamp(-cfg.epsilon, cfg.epsilon)
                delta = delta.detach()
            history.append(float(objective.detach().cpu()))

        effective_delta = (mask3 * delta).detach()
        adv_blur = (blurred + effective_delta).clamp(0.0, 1.0).detach()
        adv_deblur = self.deblur_model(adv_blur).detach()
        with torch.no_grad():
            metrics = self._metrics(blurred, deblurred, adv_blur, adv_deblur, ref_post_indices, ref_pre_indices)
        return {
            "blurred": blurred.detach(),
            "deblurred": deblurred.detach(),
            "adv_blurred": adv_blur,
            "adv_deblurred": adv_deblur,
            "delta": (adv_blur - blurred).detach(),
            "mask": mask.detach(),
            "history": torch.tensor(history),
            "metrics": metrics,
        }

    def _metrics(
        self,
        blurred: torch.Tensor,
        deblurred: torch.Tensor,
        adv_blur: torch.Tensor,
        adv_deblur: torch.Tensor,
        ref_post_indices: torch.Tensor,
        ref_pre_indices: torch.Tensor,
    ) -> dict[str, float]:
        pred_blur = self.detector.raw(blurred)
        pred_adv_blur = self.detector.raw(adv_blur)
        pred_deblur = self.detector.raw(deblurred)
        pred_adv_deblur = self.detector.raw(adv_deblur)
        pre_ref = self.detector.gathered_confidence(pred_blur, ref_pre_indices).mean()
        pre_adv = self.detector.gathered_confidence(pred_adv_blur, ref_pre_indices).mean()
        post_ref = self.detector.gathered_confidence(pred_deblur, ref_post_indices).mean()
        post_adv = self.detector.gathered_confidence(pred_adv_deblur, ref_post_indices).mean()
        pre_gap = (pre_ref - pre_adv).clamp_min(0.0)
        post_gap = (post_ref - post_adv).clamp_min(0.0)
        car = post_gap / (pre_gap + 1e-6)
        return {
            "pre_conf_ref": float(pre_ref.cpu()),
            "pre_conf_adv": float(pre_adv.cpu()),
            "post_conf_ref": float(post_ref.cpu()),
            "post_conf_adv": float(post_adv.cpu()),
            "pre_gap": float(pre_gap.cpu()),
            "post_gap": float(post_gap.cpu()),
            "cascade_amplification_ratio": float(car.cpu()),
            "l_inf": float((adv_blur - blurred).abs().max().cpu()),
            "l1": float((adv_blur - blurred).abs().mean().cpu()),
            "deblur_l1_change": float((adv_deblur - deblurred).abs().mean().cpu()),
        }


class EnsembleDeblurAmplifiedAttack(RegionAwareDeblurAmplifiedAttack):
    """One shared perturbation optimized against multiple deblur modules."""

    def __init__(
        self,
        detector: YOLOv5Detector,
        deblur_models: Mapping[str, torch.nn.Module],
        config: DeblurAmplifiedAttackConfig | None = None,
    ):
        if not deblur_models:
            raise ValueError("EnsembleDeblurAmplifiedAttack needs at least one deblur model.")
        models = dict(deblur_models)
        super().__init__(detector, next(iter(models.values())), config)
        self.deblur_models = models

    def _frequency_guided(self, grad: torch.Tensor) -> torch.Tensor:
        h, w = grad.shape[-2:]
        hints = []
        for model in self.deblur_models.values():
            if not hasattr(model, "amplification_hint"):
                continue
            hint = model.amplification_hint(h, w, grad.device)
            if hint is not None:
                hints.append(hint.to(device=grad.device, dtype=grad.dtype))
        if not hints:
            return grad
        hint = torch.stack(hints).mean(dim=0)
        while hint.ndim < grad.ndim:
            hint = hint.unsqueeze(0)
        fft = torch.fft.fft2(grad)
        return torch.fft.ifft2(fft * (1.0 + hint)).real

    def _ensemble_mask(self, blurred: torch.Tensor, refs: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        if not self.config.use_region_mask:
            return torch.ones(
                (blurred.shape[0], 1, blurred.shape[2], blurred.shape[3]),
                device=blurred.device,
                dtype=blurred.dtype,
            )
        masks = [self.build_region_mask(blurred, ref["deblurred"]) for ref in refs.values()]
        return torch.stack(masks).amax(dim=0).clamp(0.0, 1.0).detach()

    def _initial_delta_ensemble(
        self,
        blurred: torch.Tensor,
        refs: dict[str, dict[str, torch.Tensor]],
        ref_pre_indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.init_mode == "zero":
            return torch.zeros_like(blurred)
        if self.config.init_mode == "random":
            return torch.empty_like(blurred).uniform_(-self.config.epsilon, self.config.epsilon)

        post_grads = []
        for name, model in self.deblur_models.items():
            probe = blurred.detach().clone().requires_grad_(True)
            post = model(probe)
            post_obj = self.detector.hide_objective(self.detector.raw(post), refs[name]["indices"])
            post_grads.append(torch.autograd.grad(post_obj, probe, retain_graph=False, create_graph=False)[0])
        g_post = torch.stack(post_grads).mean(dim=0)

        probe_pre = blurred.detach().clone().requires_grad_(True)
        pre_obj = self.detector.hide_objective(self.detector.raw(probe_pre), ref_pre_indices)
        g_pre = torch.autograd.grad(pre_obj, probe_pre, retain_graph=False, create_graph=False)[0]

        guided = self._remove_projection(g_post, g_pre)
        guided = self._frequency_guided(guided)
        return self.config.epsilon * torch.sign(guided * (0.2 + 0.8 * mask))

    @staticmethod
    def _mean_metrics(per_deblur: dict[str, dict[str, torch.Tensor | dict[str, float]]]) -> dict[str, float]:
        metrics = [item["metrics"] for item in per_deblur.values()]
        if not metrics:
            return {}
        keys = metrics[0].keys()
        return {f"mean_{key}": float(sum(float(m[key]) for m in metrics) / len(metrics)) for key in keys}

    def run(self, blurred: torch.Tensor) -> dict[str, torch.Tensor | dict]:
        cfg = self.config
        blurred = blurred.detach().to(self.detector.device).clamp(0.0, 1.0)
        refs: dict[str, dict[str, torch.Tensor]] = {}
        with torch.no_grad():
            pred_pre_ref = self.detector.raw(blurred)
            ref_pre_indices = self.detector.topk_indices(pred_pre_ref, topk=cfg.topk)
            for name, model in self.deblur_models.items():
                deblurred = model(blurred).detach()
                pred_post_ref = self.detector.raw(deblurred)
                refs[name] = {
                    "deblurred": deblurred,
                    "indices": self.detector.topk_indices(pred_post_ref, topk=cfg.topk),
                }

        mask = self._ensemble_mask(blurred, refs)
        mask3 = mask.repeat(1, blurred.shape[1], 1, 1)
        delta = self._initial_delta_ensemble(blurred, refs, ref_pre_indices, mask).detach()
        delta = delta.clamp(-cfg.epsilon, cfg.epsilon)

        history: list[float] = []
        for _ in range(cfg.steps):
            delta = delta.detach().requires_grad_(True)
            effective_delta = mask3 * delta
            adv_blur = (blurred + effective_delta).clamp(0.0, 1.0)

            pred_pre = self.detector.raw(adv_blur)
            pre_penalty = self.detector.confidence_drop_penalty(
                pred_pre,
                pred_pre_ref,
                ref_pre_indices,
                tolerance=cfg.dormancy_tolerance,
            )
            visual = effective_delta.abs().mean()
            background = ((1.0 - mask3) * delta.abs()).mean()
            tv = total_variation(effective_delta)

            model_objectives = []
            for name, model in self.deblur_models.items():
                adv_deblur = model(adv_blur)
                deblurred = refs[name]["deblurred"]
                pred_post = self.detector.raw(adv_deblur)
                attack_obj = self.detector.hide_objective(pred_post, refs[name]["indices"])

                masked_change = (mask * (adv_deblur - deblurred).abs()).sum() / mask.sum().clamp_min(1.0)
                edge_change = edge_difference(adv_deblur, deblurred, mask=mask)
                input_change = (mask * (adv_blur - blurred).abs()).sum() / mask.sum().clamp_min(1.0)
                amplification = masked_change / input_change.clamp_min(1e-6)
                model_objectives.append(
                    attack_obj
                    + cfg.beta_deblur * masked_change
                    + cfg.beta_edge * edge_change
                    + cfg.beta_amp * amplification
                )

            objective = (
                torch.stack(model_objectives).mean()
                - cfg.lambda_pre * pre_penalty
                - cfg.lambda_vis * visual
                - cfg.lambda_bg * background
                - cfg.lambda_tv * tv
            )
            grad = torch.autograd.grad(objective, delta, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                delta = delta + cfg.step_size * torch.sign(grad)
                delta = delta.clamp(-cfg.epsilon, cfg.epsilon)
                delta = delta.detach()
            history.append(float(objective.detach().cpu()))

        effective_delta = (mask3 * delta).detach()
        adv_blur = (blurred + effective_delta).clamp(0.0, 1.0).detach()
        per_deblur: dict[str, dict[str, torch.Tensor | dict[str, float]]] = {}
        with torch.no_grad():
            for name, model in self.deblur_models.items():
                deblurred = refs[name]["deblurred"].detach()
                adv_deblur = model(adv_blur).detach()
                per_deblur[name] = {
                    "deblurred": deblurred,
                    "adv_deblurred": adv_deblur,
                    "metrics": self._metrics(
                        blurred,
                        deblurred,
                        adv_blur,
                        adv_deblur,
                        refs[name]["indices"],
                        ref_pre_indices,
                    ),
                }
        return {
            "blurred": blurred.detach(),
            "adv_blurred": adv_blur,
            "delta": (adv_blur - blurred).detach(),
            "mask": mask.detach(),
            "history": torch.tensor(history),
            "per_deblur": per_deblur,
            "metrics": self._mean_metrics(per_deblur),
        }
