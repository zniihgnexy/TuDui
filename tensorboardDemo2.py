from torch.utils.tensorboard import SummaryWriter
import torch
from pil import Image
import numpy as np

writer = SummaryWriter('logs')

image_path = 'data\\train\\ants_image\\0013035.jpg'
img_PIL = Image.open(image_path)

# 当前的类型是pil的image
# print(type(img))

# 转换成numpy的array
img_array = np.array(img_PIL)
# print(type(img))

# HWC: height, width, channel
# width can show the training process
writer = add_image("test", img_array, 2, dataformats="HWC")

for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()