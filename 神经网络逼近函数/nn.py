import numpy as np

# 神经网络结构以及功能函数
class NeuralNetwork:
    # 初始化，构造函数
    def __init__(self, layers, alpha=0.1):
        self.W = [] # weights矩阵
        self.layers = layers
        self.alpha = alpha # 学习率
        # print(len(layers))
        # 初始化Weight
        for i in np.arange(0, len(layers) - 2): # i=0
            # 先初始化前半部分的weights矩阵
            w1 = np.random.randn(layers[i] + 1, layers[i + 1] + 1) # 括号内为生成矩阵的行，列，每个元素为[0,1]间随机值
            # 归一化
            self.W.append(w1 / np.sqrt(layers[i]))
        print("未加入偏置的weight矩阵:\n", self.W)
        # 使用bias trick也就是在W矩阵最后一列加入新的一列作为bias然后weight和bias合并为一个大W矩阵
        # biases可以作为学习参数进行学习
        # 加入偏置，+1是因为有偏置单元(常数)
        w2 = np.random.randn(layers[-2] + 1, layers[-1])
        # 归一化，把这个矩阵也加入列表，[3*3,3*1]，左边的是1、2层间的weights，右边的是2、3层间的weights
        self.W.append(w2 / np.sqrt(layers[-2]))
        print("加入偏置的weight矩阵:\n", self.W)

    # 重载python的magic函数
    def __repr__(self):
        return "NeuralNetwork:{}".format("-".join(str(l) for l in self.layers))

    # 激活函数
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    # 对sigmoid函数求导
    def sigmoid_deriv(self, x):
        '''
        y = 1.0 / (1 + np.exp(-x))
        return y * (1 - y)
        '''
        return x * (1 - x)

    # 拟合，X是输入数据，y是目标结果数据，epochs迭代次数，step是步长
    def fit(self, X, y, epochs=1000, step=100):
        X = np.c_[X, np.ones((X.shape[0]))] # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，X.shape[0]返回行数（纵向）
        losses = []
        # 根据每一层网络进行反向传播，然后更新W
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target) # 更新weights
            # 控制显示，每step次迭代就计算一次loss
            if epoch == 0 or (epoch + 1) % step == 0:
                loss = self.calculate_loss(X, y)
                losses.append(loss)
                print("epoch={}, loss={:.7f}".format(epoch + 1, loss))
        return losses

    # 链式求导，x是输入数据，y是目标结果数据
    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)] # np.atleast_2d()函数用于将输入视为至少具有两个维度的数组
        # 计算这个w矩阵下整个神经网络的输出
        for layer in np.arange(0, len(self.W)): #２
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)

        # 反向传播(Back Propagation, BP)
        error = A[-1] - y # 最后一层的输出减去实际结果

        D = [error * self.sigmoid_deriv(A[-1])]
        # 反向
        for layer in np.arange(len(A)- 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        D = D[::-1] # 倒序
        # 更新权值W
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    # 预测
    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)
        # 加入偏置的情况
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
        # 正常的前向传播得到预测的输出值
        for layer in np.arange(0, len(self.W)):
            # print(np.dot(p, self.W[layer]))
            p = self.sigmoid(np.dot(p, self.W[layer])) # np.dot矩阵乘法，点乘
            print(p)
        return p


    # 计算loss，就是计算均方误差
    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss


if __name__ == '__main__':
    nn = NeuralNetwork([2, 2, 1])
    print(nn)
