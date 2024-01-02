import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./dataset_C', train=False, transform = torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

class TuDui(nn.Module):
    def __init__(self):
        super(TuDui, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x
    
tudui = TuDui()
# print(tudui)

writer = SummaryWriter('./logs')
step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    # print(output.shape)
    # print(imgs.shape)
    # print(output.shape)
    
    writer.add_images('input', imgs, step)
    
    # this will show error， 处理之后有6个通道，但是输入只有三个，所以会报错
    # writer.add_images('output', output, step)
    torch.reshape(output,(-1, 3, 30, 30))
    
    step += 1

writer.close()