# 数据集使用笔记

## 数据

data：垃圾
dataset：获取数据并获得其label，其实整理了一下，堆在了一起
    - 如何获得每个数据及其label
    - 告诉我们总共有多少个数据
dataloader：为后面网络提供不同的数据形式，对可回收垃圾进行打包，一个一个送到垃圾站

## dataset
import dataset 可以直接继承这个官方的dataset类
help(Dataset) # 获得官方解释
Dataset?? # 更方便的还有class说明和部分函数的说明，推荐这个

## 怎么给数据
code:
        # 1. Initialize file path or list of file names.
        img_path = 'E:\\learn_deep\\littleTuDui\\dataset\\train\\ants\\0013035.jpg'
        img = Image.open(img_path)
        dir_path = 'dataset/train/ants'
        img_path_list = os.listdir(dir_path)
        img.show(img_path_list[0])