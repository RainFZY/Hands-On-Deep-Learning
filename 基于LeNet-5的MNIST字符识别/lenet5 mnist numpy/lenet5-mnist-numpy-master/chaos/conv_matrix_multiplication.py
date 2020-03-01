import numpy as np


def im2col(input_size, channel):
    img = np.random.randint(0, 5, size=[input_size, input_size])  # 生成214*214的图像，每个点的取值从0-5随机生成
    print(img)
    print("输入图像维度：", img.shape)
    column = img.reshape(input_size*input_size,1)
    return column

def make_kernel_matrix(kernel_size, input_size):
    kernel = np.random.randint(0, 5, size=[kernel_size, kernel_size])  # 生成3*3的滤波器，每个点的取值从0-5随机生成
    print("滤波器：", kernel)
    print("滤波器维度：", kernel.shape)
    out_H = kernel_size
    out_W = input_size * input_size

column = im2col(4,3)
print(column)
print(column.shape)

