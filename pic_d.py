
import time

import torch
from PIL import Image
import numpy as np
from torchvision.transforms import Compose
from torchvision import transforms
import cv2
# from huggingface_hub import hf_hub_download
# from safetensors.torch import load_file

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
# checkpoint_path = "./model/model_small.safetensors"
checkpoints = 'CNN2'
# model_path = checkpoints+"/best_model.pth"
model_path = checkpoints+"/last.pth"
# model_path = "test/MS_80e.pth"
model = torch.load(model_path, weights_only=False)


outx = 200
outy = 200

resize_transform = transforms.Resize(
    size=(outx, outy),         # 目标尺寸
    interpolation=Image.BILINEAR  # 插值方法
)


# 处理函数
def process_frame(image):
    # 转换颜色空间并调整尺寸
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = frame_rgb.shape[:2]
    # print(original_height)
    # # 转换为模型输入格式
    input_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).float().to(device)
    input_tensor = resize_transform(input_tensor)
    input_tensor = input_tensor.unsqueeze(0) / 255.0  # 添加batch维度并归一化
    # print(input_tensor.shape)
    # 执行推理
    input_tensor = torch.transpose(input_tensor, 2, 3)
    with torch.no_grad():
        depth = model(input_tensor)
    depth = torch.transpose(depth, 2, 3)
    # 后处理
    # print(depth.shape)
    depth_np = depth.squeeze().cpu().numpy()
    # depth_np = depth_np.reshape(120,120)
    # print(depth_np.shape)
    # print(depth_np.min(), depth_np.max())
    depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255
    # depth_normalized = (depth_np.max() - depth_np) / (depth_np.max() - depth_np.min()) * 255
    depth_normalized = depth_normalized.astype(np.uint8)
    # print(depth_np)
    # 应用颜色映射并恢复原始尺寸
    # depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    # depth_colored = cv2.resize(depth_colored, (original_width, original_height))
    # depth_colored = cv2.resize(depth_normalized, (original_width, original_height))
    depth_colored = cv2.resize(depth_normalized, (640, 480))
    return depth_colored

dir_in = "./pic_demo/IN/"
dir_out = "./pic_demo/OUT01/"

# 处理帧并显示结果
for p in range(16):
    img = Image.open(dir_in+f"{p}.png")
    frame = np.array(img)
    depth_frame = process_frame(frame)
    # depth_frame = np.array(depth_frame)
    depth_frame = Image.fromarray(depth_frame)
    # print(depth_frame.shape)
    depth_frame.save(dir_out+f"{p}.png")
    
    