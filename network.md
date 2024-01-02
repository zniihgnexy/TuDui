# nn.network

## containers
6 moduels are defined in `nn.network.containers`:
Module:
''' python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
'''
forward function is the output of the module.
in the example above, x after the first conv layer is passed to the second conv layer. the output of the second conv layer is the output of the module.
卷积 + relu + 卷积 + relu = output

## convluational layers

卷积：2d就直接算
池化：maxpooling取最大值，avgpooling取平均值
边缘的时候，看那个ceil和floor，ceil为true的时候保留，为false的时候不保留
ceil向上取整，floor向下取整
空洞卷积：隔着几个进行卷积