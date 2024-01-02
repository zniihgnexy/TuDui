import torch
import torch.nn as nn
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]], dtype=torch.float32)

# the numbers of input channels and output channels are both 1
# 5, 5 is the size of input
# one lernel means one channel
input = torch.reshape(input, (1, 1, 5, 5))
# 1, 1 is the number of input and output channels
# 3, 3 is the size of kernel
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)

output = F.conv2d(input, kernel, stride=1, padding=0)
print(output)

# padding=1 means add 1 row and 1 column of 0 to the input
# if input is 5, 5, then the output is 7, 7
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)

# in_channels (int) – Number of channels in the input image
# out_channels (int) – Number of channels produced by the convolution