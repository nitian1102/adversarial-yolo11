import cv2
import torch
import numpy as np
from ultralytics import YOLO


class YOLOv11MotionBlurAttack:
    def __init__(self, model, kernel_size=21, angle_range=30, strength=0.4):
        self.model = model
        self.kernels = self._create_rotational_kernels(kernel_size, angle_range)
        self.strength = strength

    def _create_rotational_kernels(self, size, angle_range):
        kernels = []
        for angle in range(-angle_range, angle_range, 5):
            kernel = self._generate_motion_kernel(size, angle)
            kernels.append(kernel)
        return torch.stack(kernels)

    def _generate_motion_kernel(self, size, angle):
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        dx = np.cos(np.deg2rad(angle))
        dy = np.sin(np.deg2rad(angle))
        for i in range(size):
            x = int(center + (i - center) * dx)
            y = int(center + (i - center) * dy)
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0
        kernel /= kernel.sum()
        return torch.from_numpy(kernel).view(1, 1, size, size)

    def generate(self, x):
        grad = self._compute_gradient(x)
        batch_size = x.shape[0]
        device = x.device
        kernel_effects = []
        for kernel in self.kernels.to(device):
            convolved = torch.nn.functional.conv2d(
                grad.sign(),
                kernel.expand(x.shape[1], -1, -1, -1),
                padding=kernel.shape[-1] // 2,
                groups=x.shape[1]
            )
            kernel_effects.append(convolved.abs().sum(dim=(1, 2, 3)))
        effect_scores = torch.stack(kernel_effects, dim=1)
        best_kernel_idx = effect_scores.argmax(dim=1)
        perturbation = torch.zeros_like(x)
        for b in range(batch_size):
            selected_kernel = self.kernels[best_kernel_idx[b]].to(device)
            perturbation[b] = torch.nn.functional.conv2d(
                grad[b:b + 1].sign(),
                selected_kernel.expand(x.shape[1], -1, -1, -1),
                padding=selected_kernel.shape[-1] // 2,
                groups=x.shape[1]
            )
        adv_x = x + self.strength * perturbation
        return torch.clamp(adv_x, 0, 1)

    def _compute_gradient(self, x):
        x.requires_grad_(True)
        raw_outputs = self.model(x)
        obj_scores = raw_outputs[..., 4]
        cls_scores = raw_outputs[..., 5:]
        box_params = raw_outputs[..., :4]
        loss_components = {
            'obj_loss': -obj_scores.sigmoid().mean(),
            'cls_confusion': cls_scores.softmax(dim=-1).mean(dim=(1, 2)).sum(),
            'box_disruption': box_params.std(dim=(1, 2)).mean()
        }
        total_loss = sum(loss_components.values())
        total_loss.backward()
        return x.grad.data


def predict_video(video_path: str, model_path: str, output_path: str = "output/predicted_video.mp4"):
    """Run predictions on a video and save the result."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    model = YOLO(model_path)

    print(f"Processing video for predictions: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = model.predict(frame, imgsz=(640, 640))[0]
        result_frame = result.plot(line_width=2)
        out.write(result_frame)  # Convert RGB to BGR

    cap.release()
    out.release()
    print(f"Predicted video saved to: {output_path}")