import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import numpy as np
import os
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

# trainset
test_data = torchvision.datasets.CIFAR10(root='./dataset_C', train=False, transform=torchvision.transforms.ToTensor(), download=True)

# dataloader(batchsize=4)
# img0, target0 = test_data[0]
# img1, target1 = test_data[1]
# img2, target2 = test_data[2]
# img3, target3 = test_data[3]
# imgs, targets = test_data[0:4], batchsize=4
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中的第一张图片及其标签
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter('dataloader')
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images('Epoch: {}'.format(epoch), imgs, step)
        step += 1

writer.close()