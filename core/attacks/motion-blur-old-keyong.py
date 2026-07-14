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
    def __init__(self):
        super().__init__()

    def forward(self, x, strength):
        clamped_strength = torch.clamp(strength, -5.0, 5.0)  # 限制参数范围
        kernel_size = self._dynamic_kernel_size(clamped_strength)
        # kernel_size = self._dynamic_kernel_size(strength)
        kernel = self._build_kernel(kernel_size)
        return F.conv2d(x, kernel, padding='same', groups=3)

    def _dynamic_kernel_size(self, strength):
        """动态核尺寸计算（保留连续性）"""
        return 3 + 12 * torch.sigmoid(strength)  # 3-15范围

    def _build_kernel(self, kernel_size):
        """构建动态模糊核（修复维度问题）"""
        # 解耦计算图获取基础尺寸
        base_size = int(torch.floor(kernel_size))
        frac = kernel_size - base_size

        # 生成相邻尺寸核
        kernel_base = self._create_kernel(base_size)
        kernel_next = self._create_kernel(base_size + 1)

        # 统一尺寸（关键修复点）
        max_h = max(kernel_base.shape[-2], kernel_next.shape[-2])  # 高度
        max_w = max(kernel_base.shape[-1], kernel_next.shape[-1])  # 宽度

        # 双向填充（高度+宽度）
        kernel_base = F.pad(kernel_base,
                            (0, max_w - kernel_base.shape[-1],  # 右填充
                             0, max_h - kernel_base.shape[-2]))  # 下填充
        kernel_next = F.pad(kernel_next,
                            (0, max_w - kernel_next.shape[-1],
                             0, max_h - kernel_next.shape[-2]))

        # 尺寸一致性断言
        assert kernel_base.shape == kernel_next.shape, \
            f"核尺寸不匹配: {kernel_base.shape} vs {kernel_next.shape}"

        # 线性插值（保留梯度）
        kernel = (1 - frac) * kernel_base + frac * kernel_next

        # 扩展为3通道 [3,1,H,W]
        return kernel.repeat(3, 1, 1, 1)

    def _create_kernel(self, size):
        """创建基础核（确保数值稳定性）"""
        size = max(3, min(int(size), 15))  # 强制限制3-15
        kernel = torch.zeros((1, 1, size, size), device=device)
        center = size // 2
        kernel[..., center, :] = 1.0 / size
        return kernel

# 攻击优化器（完整实现）
class AdversarialOptimizer:
    def __init__(self, model, img_path, img_size=640):
        self.model = model
        self.original_img = self._load_image(img_path, img_size)
        self.generator = MotionBlurGenerator().to(device)
        # self.strength = nn.Parameter(torch.tensor(5.0, device=device))
        self.strength = nn.Parameter(torch.tensor(1.0, device=device))  # 初始值改为0
        # self.optimizer = optim.Adam([self.strength], lr=0.5)
        self.optimizer = optim.Adam([self.strength], lr=0.1, weight_decay=1e-4)  # 添加正则化
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer,
        #     mode='min',     # 监控loss的下降
        #     factor=0.5,     # 学习率衰减系数
        #     patience=5,     # 5次不下降则调整
        #     verbose=True    # 打印调整信息
        # )
        self.scaler = GradScaler()  # 混合精度

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

    def attack_step(self,epoch):
        """完整的攻击步骤（梯度安全）"""
        self.optimizer.zero_grad()

        # 混合精度前向传播
        with autocast():
            # 生成对抗样本（保留梯度）
            adv_img = self.generator(self.original_img, self.strength)
            result = self.model.predict(adv_img)[0]  # 直接访问网络原始输出
            # 模型推理（保持梯度连接）
            # raw_output = self.model.predict(adv_img)[0].boxes.conf  # 直接访问网络原始输出
            if len(result) > 0:
                raw_output = result.boxes.conf  # 所有检测框的置信度张量
            else:
                raw_output = torch.zeros(0)
            print(f"原始输出: {result.boxes.conf}")
            # 计算可微分损失
            loss = self._compute_loss(raw_output,epoch)

        # 梯度传播（带缩放）
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        current_lr = self.optimizer.param_groups[0]['lr']
        if epoch % 10 == 0:
            print(f"Current LR: {current_lr:.6f}")
        # 显存清理
        torch.cuda.empty_cache()

        return loss.item()

    def _compute_loss(self, raw_output,epoch):
        """可微分损失计算（关键修复）"""
        # conf_scores = torch.sigmoid(raw_output[..., 4])  # 使用sigmoid处理原始输出
        if raw_output.numel() == 0:  # 如果 raw_output 是空的
            print("raw_output is empty, skipping loss computation.")
            return torch.tensor(0.0, device=device)  # 返回默认损失值

        else:
            conf_scores = raw_output  # 假设原始输出已经是置信度
            print(f"当前置信度: {conf_scores}")
            print(f"平均置信度: {conf_scores.mean().item():.4f}")

            # 取所有anchor中最大的置信度
            max_conf = torch.max(conf_scores)

            # 动态目标：将最高置信度压制到阈值以下
            target_threshold = 0.6
            max_conf_loss = F.relu(max_conf - target_threshold)

            # 强度正则项（控制模糊程度）
            blur_penalty = 0.1 * torch.abs(self.strength)  # 增大正则化系数
            if epoch % 10 == 0:
                print(f"模糊强度: {self.strength.item():.2f} | 正则化损失: {blur_penalty.item():.4f}")
                print(f"最大置信度: {max_conf.item():.4f} | 目标阈值: {target_threshold:.2f} | 最大置信度损失: {max_conf_loss.item():.4f}")

            return max_conf_loss + blur_penalty


# 主流程（带验证）
def main():
    # 初始化组件
    model = YOLO('E:/tt100k14/weights/best.pt').to(device).eval()
    attacker = AdversarialOptimizer(
        model=model,
        img_path=r"E:\Tsinghua Data\tt100k_2021-yolo\tt100k_yolo\images\143.jpg",
        img_size=2048
    )
    losses = []
    # 训练循环
    for epoch in range(100):
        loss = attacker.attack_step(epoch)
        losses.append(loss)
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