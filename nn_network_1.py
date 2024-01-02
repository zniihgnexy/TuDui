import torch
import torch.nn as nn
import torch.nn.functional as F

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
    
    def forward(self, x):
        x = x + 1
        return x
    
tudui = Tudui()
# input is a tensor
x = torch.tensor(1.0)
output = tudui(x)
print(output)