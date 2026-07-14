import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决OpenMP警告
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 可微分运动模糊层
class MotionBlurGenerator(nn.Module):
    def __init__(self, max_kernel_size=31):  # 可配置最大尺寸
        super().__init__()
        self.max_kernel_size = max_kernel_size
        self.size_offset = 3  # 最小尺寸
        self.weight_spread = nn.Parameter(torch.tensor(0.5, device=device))  # 控制权重分布


    def forward(self, x, strength):
        # clamped_strength = torch.clamp(strength, -5.0, 5.0)  # 限制参数范围
        # kernel_size = self._dynamic_kernel_size(strength)
        kernel = self._build_kernel(strength)
        return F.conv2d(x, kernel, padding='same', groups=3)

    def _dynamic_kernel_size(self, strength):
        """动态核尺寸公式（扩展范围）"""
        # strength范围映射到 [0, max_kernel_size - size_offset]
        return self.size_offset + (self.max_kernel_size - self.size_offset) * torch.sigmoid(strength)

    def _build_kernel(self, strength):
        """生成水平可调模糊核"""
        # 动态核长度计算（保持连续性）
        base_length = 3 + 12 * torch.sigmoid(strength)  # 3~15
        spread = 1.0 + 4.0 * torch.sigmoid(self.weight_spread)  # 分散度[1,5]

        # 创建基础水平核
        kernel = self._create_horizontal_kernel(base_length, spread)
        return kernel.repeat(3, 1, 1, 1)  # 扩展至3通道

    def _create_horizontal_kernel(self, length, spread):
        """支持任意尺寸的水平核"""
        # 动态计算实际核长度
        real_length = min(int(torch.floor(length)), self.max_kernel_size)
        frac = length - real_length

        # 生成相邻尺寸的核
        kernel_base = self._base_horizontal_kernel(real_length, spread)
        kernel_next = self._base_horizontal_kernel(real_length + 1, spread)

        # 尺寸对齐
        max_w = max(kernel_base.shape[-1], kernel_next.shape[-1])
        kernel_base = F.pad(kernel_base, (0, max_w - kernel_base.shape[-1]))
        kernel_next = F.pad(kernel_next, (0, max_w - kernel_next.shape[-1]))

        # 线性插值
        return (1 - frac) * kernel_base + frac * kernel_next

    def _base_horizontal_kernel(self, length, spread):
        """基础核生成（移除硬性限制）"""
        x = torch.arange(length, device=device).float()
        center = (length - 1) / 2.0
        weights = torch.exp(-((x - center) ** 2) / (2 * (spread ** 2)))
        return weights.view(1, 1, 1, -1) / weights.sum()

# 攻击优化器（完整实现）
class AdversarialOptimizer:
    def __init__(self, model, img_path, img_size=640):
        self.model = model
        self.original_img = self._load_image(img_path, img_size)
        self.generator = MotionBlurGenerator(max_kernel_size=63).to(device)
        # self.strength = nn.Parameter(torch.tensor(5.0, device=device))
        self.strength = nn.Parameter(torch.tensor(10.0, device=device))  # 初始值改为0
        # self.optimizer = optim.Adam([self.strength], lr=0.5)
        self.optimizer = optim.Adam([self.strength], lr=5.0,betas=(0.9, 0.999))  # 添加正则化
        self.scaler = GradScaler()  # 混合精度
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer, mode='min', factor=0.5, patience=5
        )
    def _load_image(self, path, target_size):
        """加载并优化显存（关键修改）"""
        img = Image.open(path).convert('RGB')

        # 保持长宽比的缩放
        w, h = img.size
        scale = target_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.BILINEAR)

        tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        return tensor.detach().requires_grad_(False)

    def attack_step(self,imgsize):
        """完整的攻击步骤（梯度安全）"""
        self.optimizer.zero_grad()

        # 混合精度前向传播
        with autocast():
            # 生成对抗样本（保留梯度）
            adv_img = self.generator(self.original_img, self.strength)
            # results = self.model.predict(adv_img,imgsz=imgsize)  # 直接访问网络原始输出
            # if len(results) > 0:
            #     raw_output = results[0].boxes.conf  # 所有检测框的置信度张量
            # else:
            #     raw_output = torch.zeros(0)  # 空张量处理
            # print(raw_output)
            with torch.enable_grad():  # 强制启用梯度
                outputs = self.model.model(adv_img)[0]  # 原始输出

            # 提取置信度（示例为YOLOv8输出结构）
            # raw_output = outputs[0][..., 4]  # 假设第4位为置信度
            # conf_scores = torch.sigmoid(raw_output)

            # 计算损失
            loss = self._compute_loss(outputs)

        # 梯度监控
        print(f"参数梯度前: strength={self.strength.item():.2f}",
              f"梯度方向: {self.strength.grad}")
        # 梯度传播（带缩放）
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # 参数约束
        with torch.no_grad():
            self.strength.data.clamp_(min=0.0, max=20.0)  # 限制参数范围
        # 添加参数值监控
        print(f"更新后: strength={self.strength.item():.2f}",
              f"当前模糊核尺寸: {self.generator._dynamic_kernel_size(self.strength).item():.1f}")
        # torch.cuda.empty_cache()

        return loss.item()

    def _compute_loss(self, raw_output):
        """聚焦最大置信度的攻击策略"""
        # 假设输出形状为 [batch, anchors, 5+nc]
        conf_scores = torch.sigmoid(raw_output[..., 4])  # 使用sigmoid处理原始输出
        # conf_scores = raw_output
        print(f"当前置信度分数: {conf_scores}")

        # 取所有anchor中最大的置信度
        max_conf = torch.max(conf_scores)

        # 动态目标：将最高置信度压制到阈值以下
        target_threshold = 0.6
        max_conf_loss = F.relu(max_conf - target_threshold)
        print(f"当前最大置信度损失: {max_conf_loss}")

        # 强度正则项（控制模糊程度）
        blur_penalty = -0.1 * torch.abs(self.strength)
        print(f"当前模糊强度惩罚: {blur_penalty}")

        # return max_conf_loss
        return max_conf_loss + blur_penalty


# 主流程（带验证）
def main():
    imgsz = 2048
    # 初始化组件
    model = YOLO('E:/tt100k-x/weights/best.pt').to(device).eval()
    attacker = AdversarialOptimizer(
        model=model,
        img_path=r"E:\Tsinghua Data\tt100k_2021-yolo\tt100k_yolo\images\143.jpg",
        img_size=imgsz
    )
    # losses = []
    # 训练循环
    for epoch in range(100):
        loss = attacker.attack_step(imgsz)
        # losses.append(loss)
        # 动态学习率调整（添加在主循环中）
        # attacker.scheduler.step(loss)  # 关键修改点
        # 监控指标
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Strength: {attacker.strength.item():.2f}")

            # 验证梯度存在性
            assert attacker.strength.grad is not None, "梯度未生成！"
            print(f"梯度存在性验证通过，梯度均值: {attacker.strength.grad.abs().mean():.6f}")

            # 可视化
            with torch.no_grad():
                blurred = attacker.generator(
                    attacker.original_img,
                    attacker.strength
                )
                vis_img = blurred.squeeze().permute(1, 2, 0).cpu().numpy()
                # plt.imshow(vis_img)
                # plt.savefig(f"attack_{epoch}.png")
                # plt.close()
                plt.figure()
                plt.imshow(vis_img)
                plt.axis('off')  # 关闭坐标轴
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除白边
                plt.savefig(f"attack_{epoch}.png", bbox_inches='tight', pad_inches=0)
                plt.close()

if __name__ == "__main__":
    main()