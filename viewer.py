
import time

import torch
from PIL import Image
import numpy as np
from torchvision.transforms import Compose
from torchvision import transforms
import cv2
# from huggingface_hub import hf_hub_download
# from safetensors.torch import load_file

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    raise IOError("无法打开摄像头")

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

t1, t2, t3, t4 = 0, 0, 0, 0
size_h = 200
size_w = 200
ones_subarray = np.zeros((size_h, size_w, 3))
start_h = 160
start_w = 160
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
    depth_colored = cv2.resize(depth_normalized, (original_width, original_height))
    return depth_colored

fps = 0

# 主循环
try:
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        # 处理帧并显示结果
        depth_frame = process_frame(frame)
        # depth_frame = np.hstack((frame, depth_frame))
        end_time = time.time()
        processing_time = end_time - start_time
        fps = 0.1*fps + 0.9*(1/processing_time)  # 平滑处理
        cv2.putText(depth_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        cv2.imshow('RGB-Depth', depth_frame)
        # print(np.shape(frame), np.shape(depth_frame))
        # 按ESC退出
        if cv2.waitKey(1) == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()