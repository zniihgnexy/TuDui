from torchvision import transforms
from PIL import Image
import cv2
from tensorboardX import SummaryWriter

# 在python中的用法 - tensor
# totensor进行讲解，通过transforms.ToTensor()将PIL.Image或者numpy.ndarray转化为tensor
# 1. transforms如何使用
# 2. 为什么需要tensor数据类型

# 相对路径and绝对路径
# img_path = 'E:\\learn_deep\\littleTuDui\\data\\train\\ants_image\\0013035.jpg'
img_path = 'data\\train\\ants_image\\0013035.jpg'
img = Image.open(img_path)
# print(img)

writer = SummaryWriter("logs")

# 1. transforms如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

print(tensor_img)

# 2. 为什么需要tensor数据类型
cv_img = cv2.imread(img_path)
print(type(cv_img))

writer.add_image("tensor_img", tensor_img)

writer.close()