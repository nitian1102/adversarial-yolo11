from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torchvision.ops import nms


@dataclass
class DetectionSummary:
    count: int
    mean_confidence: float
    max_confidence: float


class YOLOv5Detector:
    """Gradient-capable YOLOv5s adapter.

    The default path loads ultralytics/yolov5 through torch.hub with
    autoshape disabled, so tensor inputs keep gradients.
    """

    def __init__(
        self,
        model_name: str = "yolov5s",
        weights: str | None = None,
        repo_or_dir: str = "ultralytics/yolov5",
        source: str = "github",
        device: torch.device | str | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        hub_kwargs: dict[str, Any] = {"autoshape": False, "verbose": False}
        if weights:
            weights_path = Path(weights)
            if not weights_path.exists():
                raise FileNotFoundError(f"YOLOv5 weights not found: {weights}")
            self.model = torch.hub.load(
                repo_or_dir,
                "custom",
                path=str(weights_path),
                source=source,
                trust_repo=True,
                **hub_kwargs,
            )
        else:
            self.model = torch.hub.load(
                repo_or_dir,
                model_name,
                pretrained=True,
                source=source,
                trust_repo=True,
                **hub_kwargs,
            )
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.names = getattr(self.model, "names", None)

    def raw(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.device)
        out = self.model(image)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.ndim != 3:
            raise RuntimeError(f"Unexpected YOLOv5 output shape: {tuple(out.shape)}")
        return out

    @staticmethod
    def confidences(pred: torch.Tensor) -> torch.Tensor:
        obj = pred[..., 4].clamp(0.0, 1.0)
        cls = pred[..., 5:].clamp(0.0, 1.0).amax(dim=-1)
        return obj * cls

    def topk_indices(
        self,
        pred: torch.Tensor,
        topk: int = 40,
        threshold: float | None = None,
    ) -> torch.Tensor:
        conf = self.confidences(pred).detach()
        flat = conf.flatten(start_dim=1)
        if threshold is not None:
            keep = flat >= threshold
            if keep.any(dim=1).all():
                masked = flat.masked_fill(~keep, -1.0)
            else:
                masked = flat
        else:
            masked = flat
        k = min(int(topk), masked.shape[1])
        return masked.topk(k=k, dim=1).indices

    def gathered_confidence(self, pred: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        conf = self.confidences(pred).flatten(start_dim=1)
        indices = indices.to(conf.device)
        return conf.gather(1, indices.clamp_max(conf.shape[1] - 1))

    def hide_objective(self, pred: torch.Tensor, indices: torch.Tensor | None = None, topk: int = 40) -> torch.Tensor:
        """Objective to maximize when hiding objects."""
        if indices is None:
            indices = self.topk_indices(pred.detach(), topk=topk)
        return -self.gathered_confidence(pred, indices).mean()

    def confidence_drop_penalty(
        self,
        pred_adv: torch.Tensor,
        pred_ref: torch.Tensor,
        indices: torch.Tensor,
        tolerance: float = 0.03,
    ) -> torch.Tensor:
        ref_conf = self.gathered_confidence(pred_ref.detach(), indices).mean()
        adv_conf = self.gathered_confidence(pred_adv, indices).mean()
        return F.relu(ref_conf - adv_conf - tolerance)

    def summarize_raw(self, pred: torch.Tensor, threshold: float | None = None) -> DetectionSummary:
        threshold = self.confidence_threshold if threshold is None else threshold
        conf = self.confidences(pred).detach().flatten()
        selected = conf[conf >= threshold]
        if selected.numel() == 0:
            return DetectionSummary(count=0, mean_confidence=0.0, max_confidence=0.0)
        return DetectionSummary(
            count=int(selected.numel()),
            mean_confidence=float(selected.mean().cpu()),
            max_confidence=float(selected.max().cpu()),
        )

    @staticmethod
    def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
        xy, wh = xywh[..., :2], xywh[..., 2:4]
        half = wh / 2.0
        return torch.cat([xy - half, xy + half], dim=-1)

    def detections(self, image: torch.Tensor, max_det: int = 100) -> list[dict[str, Any]]:
        with torch.no_grad():
            pred = self.raw(image)
            pred0 = pred[0]
            conf_all = self.confidences(pred0)
            cls_conf, cls_idx = pred0[:, 5:].max(dim=-1)
            keep = conf_all >= self.confidence_threshold
            if not keep.any():
                return []
            boxes = self.xywh_to_xyxy(pred0[keep, :4])
            scores = conf_all[keep]
            classes = cls_idx[keep]
            keep_nms = nms(boxes, scores, self.iou_threshold)[:max_det]
            results = []
            for idx in keep_nms:
                class_id = int(classes[idx].cpu())
                label = str(class_id)
                if self.names is not None:
                    label = self.names[class_id] if isinstance(self.names, list) else self.names.get(class_id, label)
                results.append(
                    {
                        "box": [float(v) for v in boxes[idx].detach().cpu()],
                        "score": float(scores[idx].detach().cpu()),
                        "class_id": class_id,
                        "label": label,
                    }
                )
            return results

    def prediction_box_mask(
        self,
        image: torch.Tensor,
        pred: torch.Tensor | None = None,
        topk: int = 10,
        dilation: float = 0.15,
    ) -> torch.Tensor:
        if pred is None:
            pred = self.raw(image)
        b, _, h, w = image.shape
        mask = torch.zeros((b, 1, h, w), device=image.device, dtype=image.dtype)
        pred = pred.detach()
        conf = self.confidences(pred)
        k = min(topk, conf.shape[1])
        idx = conf.topk(k=k, dim=1).indices
        for batch in range(b):
            boxes = self.xywh_to_xyxy(pred[batch, idx[batch], :4])
            for box in boxes:
                x1, y1, x2, y2 = box
                bw = (x2 - x1).abs()
                bh = (y2 - y1).abs()
                x1 = int(torch.floor((x1 - dilation * bw).clamp(0, w - 1)).item())
                y1 = int(torch.floor((y1 - dilation * bh).clamp(0, h - 1)).item())
                x2 = int(torch.ceil((x2 + dilation * bw).clamp(0, w - 1)).item())
                y2 = int(torch.ceil((y2 + dilation * bh).clamp(0, h - 1)).item())
                if x2 > x1 and y2 > y1:
                    mask[batch, :, y1:y2, x1:x2] = 1.0
        return mask
