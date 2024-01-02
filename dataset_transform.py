import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import numpy as np
import os
import cv2
from tensorboardX import SummaryWriter

dataset_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

train_set = torchvision.datasets.CIFAR10(root='./dataset_C', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset_C', train=False, transform=dataset_transform, download=True)

# print(test_set[0])
# print(test_set.classes)

# img, target = test_set[0]
# print(img)
# # target here is the index of the class
# print(target)
# print(test_set.classes[target])
# img.show()

print(test_set[0])

writer = SummaryWriter('p10')
for i in range(10):
    # img is a tensor, test_set[i] can get the i-th data in the test_set
    img, target = test_set[i]
    writer.add_image('test_set', img, i)

writer.close()