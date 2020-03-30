import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
import os
import time

# 设置当前路径
os.chdir(os.path.split(os.path.realpath(__file__))[0])

#-----------------------------------------功能函数--------------------------------------------
def loadDataset():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

# one-hot（独热）编码，每一个特征都只有一个1，其余都是0
def oneHot(Y, size_out):
    N = Y.shape[0]
    Z = np.zeros((N, size_out))
    Z[np.arange(N), Y] = 1
    return Z

def drawLossCurve(losses,interval):
    t = np.arange(len(losses))
    plt.style.use("ggplot")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(interval*t, losses) # 确保epoch的显示不受interval影响


# 在X的大小范围内随机产生一个大小为batch_size的batch
def makeBatch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N-batch_size) # 产生两个数之间一个整数型随机数
    return X[i:i+batch_size], Y[i:i+batch_size]

# MATLAB内置函数，优化卷积运算，为了在计算时读取连续的内存，减少时间
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


# im2col的逆函数，还原原矩阵
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


#----------------------------------激活函数---------------------------------------
# 对比sigmoid更具优越性
class ReLU():

    def __init__(self):
        self.cache = None

    def forward(self, X):
        out = np.maximum(0, X)
        self.cache = X
        return out

    def backward(self, outX):
        X = self.cache
        dX = np.array(outX, copy=True)
        dX[X <= 0] = 0
        return dX


# 用于最后的输出分类
class Softmax():

    def __init__(self):
        self.cache = None

    def forward(self, X):
        maxes = np.amax(X, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        Y = np.exp(X - maxes)
        Z = Y / np.sum(Y, axis=1).reshape(Y.shape[0], 1)
        self.cache = (X, Y, Z)
        return Z # distribution

    def backward(self, outX):
        X, Y, Z = self.cache
        dZ = np.zeros(X.shape)
        dY = np.zeros(X.shape)
        dX = np.zeros(X.shape)
        N = X.shape[0]
        for n in range(N):
            i = np.argmax(Z[n])
            dZ[n,:] = np.diag(Z[n]) - np.outer(Z[n],Z[n])
            M = np.zeros((N,N))
            M[:,i] = 1
            dY[n,:] = np.eye(N) - M
        dX = np.dot(outX,dZ)
        dX = np.dot(dX,dY)
        return dX
#--------------------------------------- 计算loss --------------------------------------------------
# 负对数似然估计，Negative log likelihood loss，即交叉熵损失
def NLLLoss(Y_pred, Y_true):
    loss = 0.0
    N = Y_pred.shape[0]
    M = np.sum(Y_pred*Y_true, axis=1)
    for e in M:
        #print(e)
        if e == 0:
            loss += 500
        else:
            loss += -np.log(e)
    return loss/N

# 整个计算交叉熵损失的过程
class CountLoss():
    def __init__(self):
        pass
    # 预测值，真实值
    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        softmax = Softmax()
        # softmax输出预测值属于某一类的概率
        prob = softmax.forward(Y_pred)
        # 用NLLLoss计算交叉熵损失
        loss = NLLLoss(prob, Y_true)
        Y_serial = np.argmax(Y_true, axis=1)
        outX = prob.copy()
        outX[np.arange(N), Y_serial] -= 1
        return loss, outX


#----------------------------------------优化方法---------------------------------------------------
class SGD():
    def __init__(self, params, lr=0.001, reg=0):
        self.parameters = params
        self.lr = lr
        self.reg = reg

    def update(self):
        for param in self.parameters:
            param['val'] -= (self.lr*param['grad'] + self.reg*param['val'])

# optimizer = SGD(model.get_params(), lr=0.0001, reg=0)

#----------------------------------------神经网络每一层---------------------------------------------
# 卷积层
class Conv:
    # 输入feature map数，输出feature map数（卷积核种类），输入feature map高，宽，卷积核大小，卷积核滑动步幅，填充大小
    def __init__(self, Cin, Cout, X_H, X_W, size, stride, pad):
        # 卷积核在图像的水平方向和垂直方向的滑动步长
        self.stride = stride
        # 输入图像周围各填充pad个0
        self.pad = pad
        # 中间数据（backward时使用）
        self.x = None
        self.col = None
        self.col_W = None
        # np.random.normal：返回一个正态分布，均值，标准差（分布的宽度），shape
        # shape:
        self.W = {'val': np.random.normal(0.0,np.sqrt(2/Cin),(Cout,Cin,size,size)), 'grad': 0} # 卷积核初始化
        # np.random.randn(x)从标准正态分布中返回一个1行x列的矩阵
        self.b = {'val': np.random.randn(Cout), 'grad': 0} # 输出几个feature map，就要几个不同的偏置项
        self.X_H = X_H
        self.X_W = X_W
        self.Cin = Cin

    def forward(self, x):
        # shape[0]：读取矩阵第一维度的长度
        # print(x.shape) reshape前最初始为(100,784)
        x = x.reshape(x.shape[0], self.Cin, self.X_H, self.X_W)
        # 读取数据x每一维的大小
        N, C, H, W = x.shape
        # print(x.shape) # 100,1,28,28;100,6,14,14;....
        # 输出feature map数/filter number/卷积核种类，输入feature map数，卷积核高/filter height，宽/filter width
        FN, C, FH, FW = self.W['val'].shape
        # print(self.W['val'].shape) # 6,1,5,5;16,6,5,5...

        # 计算输出feature map大小
        Ho = int((H - FH + 2*self.pad) / self.stride) + 1 # Ho=Wo=(H−F+2×P)/S+1=(高−卷积核的边长+2×图像边扩充大小)/滑动步长+1
        Wo = int((W - FW + 2*self.pad) / self.stride) + 1
        # 利用im2col转换为行
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 卷积核转换为列，展开为2维数组
        col_W = self.W['val'].reshape(FN, -1).T
        # 计算正向传播
        out = np.dot(col, col_W) + self.b['val']
        out = out.reshape(N, Ho, Wo, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, outX):
        # print(outX.shape) # 100,16,10,10; 100,6,28,28
        # 卷积核大小
        FN, C, FH, FW = self.W['val'].shape
        outX = outX.transpose(0,2,3,1).reshape(-1, FN)

        db = np.sum(outX, axis=0)
        dW = np.dot(self.col.T, outX)
        dW = dW.transpose(1, 0).reshape(FN, C, FH, FW)

        self.W['grad'] = dW
        self.b['grad'] = db

        dcol = np.dot(outX, self.col_W.T)
        # 逆转换
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


# 池化层，用最大池化，因为保留的信息越强
class MaxPool:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        # print(x.shape) #100,6,28,28;100,16,10,10
        N, C, H, W = x.shape
        Ho = int((H - self.pool_h) / self.stride + 1)
        Wo = int((W - self.pool_w) / self.stride + 1)
		# 展开
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
		# 最大值
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        # 转换
        out = out.reshape(N, Ho, Wo, C).transpose(0, 3, 1, 2)
        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, outX):
        outX = outX.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((outX.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = outX.flatten()
        dmax = dmax.reshape(outX.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx


# 全连接层，Fully connected layer
class FC():

    def __init__(self, size_in, size_out): # 输入维数，输出维数
        self.cache = None
        # self.W = {'val': np.random.randn(size_in, size_out), 'grad': 0}
        self.W = {'val': np.random.normal(0.0, np.sqrt(2/size_in), (size_in,size_out)), 'grad': 0} # 随机初始化weight
        self.b = {'val': np.random.randn(size_out), 'grad': 0} # 随机初始化bias

    def forward(self, X):
        # print(X.shape)
        out = np.dot(X, self.W['val']) + self.b['val']
        self.cache = X
        return out

    def backward(self, outX):
        # print("FC: backward")
        X = self.cache
        dX = np.dot(outX, self.W['val'].T).reshape(X.shape)
        self.W['grad'] = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, outX)
        self.b['grad'] = np.sum(outX, axis=0)
        return dX

#------------------------------------------------- 构造Lenet-5 -------------------------------------------------
class LeNet5():

    def __init__(self):
        # Mnist数据集都是是28*28，pad设置为2，给输入图像增加了一圈黑边，使输入图像大小变成了32x32，这样的目的是为了在下层卷积过程中保留更多原图的信息。
        # 这样用5*5的卷积核，pad为2，输出仍然是28*28
        self.conv1 = Conv(1, 6, 28, 28, 5, 1, 2) # 输入feature map数，输出feature map数（卷积核种类），输入feature map大小，卷积核大小，步幅，填充
        self.ReLU1 = ReLU()
        self.pool1 = MaxPool(2,2,2,0) # 采样区域高，采样区域宽，滑动步长（取2，保证所选的区域无重叠），填充大小
        self.conv2 = Conv(6, 16, 14, 14, 5, 1, 0)
        self.ReLU2 = ReLU()
        self.pool2 = MaxPool(2,2,2,0)
        self.FC1 = FC(16*5*5, 120) # 输入维数，节点数
        self.ReLU3 = ReLU()
        self.FC2 = FC(120, 84)
        self.ReLU4 = ReLU()
        self.FC3 = FC(84, 10)
        self.Softmax = Softmax()
        self.p2_shape = None
        
    def forward(self, X):
        # print("X.shape:",X.shape) (100/60000/10000, 784)
        h1 = self.conv1.forward(X)
        a1 = self.ReLU1.forward(h1)
        p1 = self.pool1.forward(a1)
        h2 = self.conv2.forward(p1)
        a2 = self.ReLU2.forward(h2)
        p2 = self.pool2.forward(a2)
        self.p2_shape = p2.shape
        fl = p2.reshape(X.shape[0],-1) # Flatten 转化为列向量
        h3 = self.FC1.forward(fl) # C5写成全连接层，其实等效于卷积层
        a3 = self.ReLU3.forward(h3)
        h4 = self.FC2.forward(a3)
        a5 = self.ReLU4.forward(h4)
        h5 = self.FC3.forward(a5)
        # a5 = self.Softmax.forward(h5)
        return h5

    def backward(self, outX):
        #outX = self.Softmax.backward(outX)
        outX = self.FC3.backward(outX)
        outX = self.ReLU4.backward(outX)
        outX = self.FC2.backward(outX)
        outX = self.ReLU3.backward(outX)
        outX = self.FC1.backward(outX)
        outX = outX.reshape(self.p2_shape) # reshape
        outX = self.pool2.backward(outX)
        outX = self.ReLU2.backward(outX)
        outX = self.conv2.backward(outX)
        outX = self.pool1.backward(outX)
        outX = self.ReLU1.backward(outX)
        
        outX = self.conv1.backward(outX)
        
    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b] = params


#---------------------------------------预处理----------------------------------------------
# 加载数据集
X_train, Y_train, X_test, Y_test = loadDataset()
X_train, X_test = X_train/float(255), X_test/float(255) # 归一化
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

print("训练集图像数量：",len(X_train)) # 60000
print("训练集标签数量：",len(Y_train))
print("测试集图像数量：",len(X_test)) # 10000
print("测试集标签数量：",len(Y_test))

batch_size = 100
size_in = 784 # 28*28图像
size_out = 10 # 数字0-9
print("batch size: ",batch_size,", size_in: ",size_in,", size_out: ",size_out)
#---------------------------------- T r a i n ----------------------------------------
def train(epoch, interval):
    print("---------- Start Training! ----------")
    trainTime_start = time.time()
    print("epoch =", epoch)
    # 每一个epoch只跑了一个batch，也就是60000张中的batch_size张图，最后测试train accuracy用完整的数据集
    losses = []

    model = LeNet5()
    optimizer = SGD(model.get_params(), lr=0.0001, reg=0) # 传入model的参数，这样保证之后优化器作用在LeNet5上，参数得到更新
    countLoss = CountLoss()

    for i in range(epoch):
        # get batch, make onehot
        X_batch, Y_batch = makeBatch(X_train, Y_train, batch_size)
        Y_batch = oneHot(Y_batch, size_out) # 交叉熵损失需要

        Y_pred = model.forward(X_batch)  # 前向传播，forward函数中的conv等函数用到之前的参数，实现一轮的参数更新、传递
        loss, outX = countLoss.get(Y_pred, Y_batch)  # 计算损失，softmax加在这
        model.backward(outX)  # 后向传播
        optimizer.update()  # 优化，更新参数

        if i % interval == 0: # 每1000/20000采样一次
            print("%s%% epoch: %s, loss: %s" % (100 * i / epoch, i, loss))
            losses.append(loss)

    # 保存参数数据
    weights = model.get_params()
    with open("weights.pkl", "wb") as f:
        pickle.dump(weights, f)  # 使用pickle模块将数据对象保存到文件
        print("模型参数保存成功！")

    # 绘制loss曲线
    drawLossCurve(losses,interval)
    plt.show()

    # 统计训练时间
    trainTime_end = time.time()
    print('total training time =', (trainTime_end - trainTime_start) / 60.0, 'min')

    return model
#------------------------------------- T e s t -----------------------------------------
# 加载上次训练完成的weights，不训练直接测试
def loadWeights():
    f = open('weights.pkl', 'rb')
    weights = pickle.load(f)
    model = LeNet5()
    model.set_params(weights)

    return model

# 获取model，model的参数用本次训练得到的或上次训练得到的都行
# model = train(10000,500)
model = loadWeights()

def all_test():
    print("---------- Start Testing! ----------")
    testTime_start = time.time()

    test_size = len(X_test)  # 测试集大小（最多10000）
    print("取", test_size, "张测试集图像，测试测试集准确率")

    # 测试集准确率
    Y_pred = model.forward(X_test)
    result = np.argmax(Y_pred, axis=1) - Y_test
    result = list(result)
    print("TEST: Correct: ", result.count(0), " out of ", X_test.shape[0], ", accuracy =",
          result.count(0) * 100 / X_test.shape[0], "%")

    # 统计测试时间
    testTime_end = time.time()
    print('total test time =', (testTime_end - testTime_start) / 60.0, 'min')


def part_test(num):
    print("---------- Start Testing! ----------")
    testTime_start = time.time()

    test_num = num
    print("取",test_num,"张训练集图像，测试训练集准确率")
    X_batch_test, Y_batch_test = makeBatch(X_train, Y_train, test_num) # 根据给定的train_size随机生成

    # 训练集准确率
    Y_pred = model.forward(X_batch_test)
    result = np.argmax(Y_pred, axis=1) - Y_batch_test
    result = list(result)
    print("labels:",Y_batch_test)
    print("prediction:",np.argmax(Y_pred, axis=1))
    print("predicting result:",result)
    print("TRAIN: Correct: ",result.count(0)," out of ", X_batch_test.shape[0], ", accuracy =", result.count(0)*100/X_batch_test.shape[0],"%")

    # 统计测试时间
    testTime_end = time.time()
    print('total test time =', (testTime_end - testTime_start) / 60.0, 'min')


# all_test()
part_test(20) # 自定义测试图片数量
