import numpy as np

# 对于卷积核每一次要处理的小窗，将其展开到新矩阵的一行（列），新矩阵的列（行）数，就是对于一副输入图像，卷积运算的次数（卷积核滑动的次数）
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    # 输入数据的形状
    # N：批数目，C：通道数，H：输入数据高，W：输入数据宽
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1  # 输出数据的高
    out_w = (W + 2 * pad - filter_w) // stride + 1  # 输出数据的长
    # 填充高、宽
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    # (N, C, filter_h, filter_w, out_h, out_w)的0矩阵
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    # 按(0, 4, 5, 1, 2, 3)顺序，交换col的列，然后改变形状
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def conv3D(img, f, channel=3, stride=1 , pad=0, num=1): # # 二维图像,二维滤波器,channel,stride,padding,滤波器个数
    img = np.pad(img, [(pad, pad), (pad, pad)], 'constant', constant_values=0) # 填充pad宽的0值黑边
    inW, inH = img.shape
    fW, fH = f.shape
    outW = int((inW - fW) / stride + 1) # 输出图像宽度
    outH = int((inH - fH) / stride + 1) # 输出图像高度
    arr = np.zeros(shape=(outW, outH))
    # 卷积
    for g in range(outH):
        for t in range(outW):
            s = 0
            for i in range(fW):
                for j in range(fH):
                    s += img[i + g * stride][j + t * stride] * f[i][j]
            arr[g][t] = s * channel # 单层卷积结果乘上channel数求和

    print("单个卷积核卷积结果：",arr)
    arr3D = arr
    for i in range(num-1):
        arr3D = np.vstack((arr3D,arr)) # 设置方向合并矩阵，但仍然是二维的
    arr3D = arr3D.reshape(num,outW,outH)
    # print(num,"个卷积核卷积结果：",arr3D)
    return arr3D


a = np.random.randint(0,5,size=[214,214]) # 生成214*214的图像，每个点的取值从0-5随机生成
print("输入图像维度：",a.shape)

filter = np.random.randint(0,5,size=[3,3]) # 生成3*3的滤波器，每个点的取值从0-5随机生成
print("滤波器：",filter)
print("滤波器维度：",filter.shape)

array = conv3D(a,filter,3,1,0,64) # 二维图像,二维滤波器,channel,stride,padding,滤波器个数
print("输出图像维度：",array.shape)


