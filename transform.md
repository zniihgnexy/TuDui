in pycharm
alt+7 能看 structure 能使直接查类的情况
ctrl按住点击对应的内容也可以直接看例子

ToTensor： 把一个image或者numpy数据类型的转为一个tensor
。。。都可以在pycharm类型查看的地方看

transform.py 文件：像一个工具箱
工具：totensor，resize

特定格式的image -> 创建自己的工具 -> 使用工具（输入输出） -> 想要的图片的结果

一般tensor的图片类型都有个backword的钩子留着，用来做bp

## 常见的transform类型
输入    PIL     Image.open()
输出    tensor  ToTensor()
作用    narrays cv.imread()

transform
compose类，可以组合不同的变换，通过查询class可以看

call函数，能调用整个类的内容

normalize类
resize类
topil:看官方类的文档

不知道返回值数据类型的时候 - 
print
print(type())
debug
自己查


## 使用数据集
CIFAR10 数据集

ctrl + P 查看当前需要的输入等等，光标在括号中