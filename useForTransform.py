import os
from PIL import Image
from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

# SummaryWriter("logs")

img = Image.open('images\\bee_sample_pic.jpg')
# print(img)

# totensor usage example
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
print(type(img_tensor))
# write.add_image("ToTensor", img_tensor)

# topilimage usage example

# normalization usage example, tensor class type

# input[channel] = (input[channel] - mean[channel]) / std[channel]
# 2*input - 1
# [0, 1] -> [-1, 1]
# the numbers in the [] are the mean and std of the three channels
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
# write.add_image("Normalize", img_norm)

# resize usage example, the input has to be the pil image
print(img.size)
trans_resize = transforms.Resize((100, 100))
# img PIL -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> img_resize tensor
img_resize = trans_totensor(img_resize)
print(type(img_resize))
print(img_resize.size)
# writer.add_image("Resize", img_resize, 0)

# compose usage example
# compose - resize - totensor - normalize
trans_resize_2 = transforms.Resize((100, 100))
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
print(type(img_resize_2))
print(img_resize_2.size)
# writer.add_image("Compose", img_resize_2, 0)
save_path = os.path.join("images", "bee_sample_pic_resize_2.jpg")
save_img = transforms.ToPILImage()(img_resize_2)
save_img.save(save_path)

# randomcrop usage example
trans_random = transforms.RandomCrop([100, 200])
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)

save_path_random = os.path.join("images", "bee_sample_pic_random.jpg")
save_img_random = trans_random(img)
save_img.save(save_path_random)

# writer.close()