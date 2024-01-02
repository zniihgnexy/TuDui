import torch
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
import torchvision

dataset = torch.vision.datasets.CIFAR10(root='./dataset_C', train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)

dataloader = DataLoader(dataset, batch_size=64)

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

# 一般不用全都给出，可以自动调整，不用自己写数据了
# kernel = torch.tensor([[1, 2, 1],
#                        [0, 1, 0],
#                        [2, 1, 0]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)

#特征降维，计算的参数变少了，有点像1080p变成了720p
class TuDui(nn.Module):
    def __init__(self):
        super(TuDui, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

tudui = TuDui()
output = tudui(input)
print(output)

SummaryWriter = torch.utils.tensorboard.SummaryWriter('./logs_maxpool')
step = 0
for data in dataloader:
    imgs, targets = data
    SummaryWriter.add_images('input', imgs, step)
    output2 = tudui(imgs)
    SummaryWriter.add_images('output', output2, step)
    step += 1

SummaryWriter.close()