# core/attacks/base_attack.py
import torch


class BaseAttack:
    """攻击方法基类"""

    def __init__(self, **params):
        self._validate_params(params)
        self.params = params

    def _validate_params(self, params):
        """参数验证模板方法"""
        required_params = self.required_params()
        missing = [p for p in required_params if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

    def required_params(self) -> list:
        """子类必须实现的必需参数列表"""
        raise NotImplementedError

    def execute(self, model, image: torch.Tensor) -> torch.Tensor:
        """执行攻击的核心方法"""
        raise NotImplementedError

    @staticmethod
    def get_predictions(model, image: torch.Tensor):
        """统一获取模型预测"""
        pred = model(image)
        return pred.boxes.data if hasattr(pred, 'boxes') else pred[0]

    @staticmethod
    def calculate_loss(predictions: torch.Tensor, targeted: bool = False) -> torch.Tensor:
        """统一损失计算"""
        obj = predictions[:, 4]
        cls_conf = predictions[:, 5:].max(1)[0]
        return (-1 if targeted else 1) * (obj.sum() + cls_conf.sum())