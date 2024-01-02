from torch.utils.tensorboard import SummaryWriter
import torch

writer = SummaryWriter('logs')

# writer.add_image()
# y = x
# 一般用来写train loss和test loss
for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()