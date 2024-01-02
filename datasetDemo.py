from torch.utils.data import Dataset
from PIL import Image
import os
import cv2

class MyData(Dataset):

    def __init__(self, root_dir, label_dir, transform=None):
        # 1. Initialize file path or list of file names.
        # img_path = 'E:\\learn_deep\\littleTuDui\\dataset\\train\\ants\\0013035.jpg'
        # img = Image.open(img_path)
        # dir_path = 'dataset/train/ants'
        # img_path_list = os.listdir(dir_path)
        # img.show(img_path_list[0])
        root_dir = 'E:\\learn_deep\\littleTuDui\\dataset\\train'
        label_dir = 'ants'
        # add the path of the label
        # path = os.path.join(root_dir, label_dir)
        # self里的后面也能用
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path_list = os.listdir(self.path)
        return

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path, self.img_path_list[index]))
        img.name = self.img_path_list[index]
        img_item_path = os.path.join(self.path, img.name)
        # 2. Read one data from file
        # index = 0
        # img_name = self.img_path_list[index]
        # img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path_list)


root_dir = 'E:\\learn_deep\\littleTuDui\\dataset\\train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

# ants_dataset.__getitem__(0)
# img, label = bees_label_dir[0]
# img.show(img)

train_dataset = ants_dataset + bees_dataset
# len(train_dataset)
# len(ants_dataset)
# len(bees_dataset)
# img, label = train_dataset[0]