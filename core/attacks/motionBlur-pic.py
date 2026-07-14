import torch
import torch.optim as optim
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
# from .base_attack import BaseAttack
from core.attacks.base_attack import BaseAttack


# 检查并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 初始化模型
model = YOLO('E:/tt100k-x/weights/best.pt').to(device).eval()
for param in model.parameters():
    param.requires_grad = False

# 定义可优化参数（示例：水平运动模糊长度）
blur_length = torch.tensor(5.0, requires_grad=True, device=device)
optimizer = optim.Adam([blur_length], lr=0.1)

# 原始图像处理
img = Image.open(r"E:\Tsinghua Data\tt100k_2021-yolo\tt100k_yolo\images\143.jpg")
original_image = transforms.ToTensor()(img).unsqueeze(0).to(device, torch.float32)  # 明确指定数据类型


def generate_kernel(length, device):
    """生成可微分运动模糊核"""
    kernel_size = int(torch.clamp(length, min=3, max=15))
    kernel = torch.zeros((3, 1, kernel_size, kernel_size),
                         device=device,
                         requires_grad=False)  # 注意这里设为False
    center = kernel_size // 2
    kernel[:, :, center, :] = 1.0 / kernel_size
    return kernel

def attack_loss(results):
    """计算需要梯度的损失"""
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            # 保持梯度传播
            conf = result.boxes.conf
            loss = loss - conf.mean()  # 最大化损失取负
    return loss

# 迭代优化流程
for epoch in range(100):
    optimizer.zero_grad()

    # 生成运动模糊核（全部在GPU上创建）
    # kernel_size = int(torch.clamp(blur_length, min=3, max=15))
    # kernel = torch.zeros((3, 1, kernel_size, kernel_size), device=device)
    # center = kernel_size // 2
    # kernel[:, :, center, :] = 1.0 / kernel_size
    kernel = generate_kernel(blur_length, device)

    # 应用卷积（确保输入输出都在GPU）
    blurred_image = torch.nn.functional.conv2d(
        input=original_image,
        weight=kernel,
        padding='same',
        groups=3
    )

    # 前向传播获取YOLO输出
    results = model(blurred_image)
    total_loss = attack_loss(results)

    # 梯度检查
    print(f"Blurred image grad: {blurred_image.requires_grad}")  # 应为True
    print(f"Loss grad_fn: {total_loss.grad_fn}")  # 应有有效梯度函数

    # 多帧结果处理（假设单张图像）
    # total_loss = torch.tensor(0.0, device=device)  # 初始化损失
    # for result in results:  # 遍历每张图像的检测结果
    #     total_loss += BaseAttack.motion_blur_loss(result, device)

###
    # 在训练循环中
    #outputs = model(blurred_image)
    # loss = -get_detection_loss(outputs)  # 负号表示要最小化置信度
    # 计算损失（假设攻击目标为降低person类的置信度）
    #pred = BaseAttack.get_predictions(model, img)
    # loss = BaseAttack.calculate_loss(pred)
    # Calculate losses
    #objectness = pred[:, 4]  # Confidence in object detection
    #class_confidence = pred[:, 5:].max(1)[0]  # Max class confidence
    #total_loss = objectness.sum() + class_confidence.sum()
###


    # 反向传播与优化
    total_loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {total_loss.item()}, Blur Length: {blur_length.item()}")