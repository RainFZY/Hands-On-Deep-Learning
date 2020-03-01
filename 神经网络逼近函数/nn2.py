import numpy as np

# 神经网络结构以及功能函数
class NeuralNetwork:
    # 初始化，构造函数
    def __init__(self, layers, alpha=0.1):
        self.W = [] # weights矩阵
        self.layers = layers
        self.alpha = alpha # 学习率
        # 初始化Weight
        # print(len(layers))
        for i in np.arange(0, len(layers) - 2): # i=0
            # 先初始化前半部分的weights矩阵
            w1 = np.random.randn(layers[i] + 1, layers[i + 1] + 1) # 括号内为生成矩阵的行，列，，每个元素为[0,1]间随机值
            # 归一化
            self.W.append(w1 / np.sqrt(layers[i]))
        # print("W without bias trick:\n", self.W)
        # 使用bias trick也就是在W矩阵最后一列加入新的一列作为bias然后weight和bias合并为一个大W矩阵
        # biases可以作为学习参数进行学习
        # 加入偏置，+1是因为有偏置单元(常数)
        w2 = np.random.randn(layers[-2] + 1, layers[-1])
        # 归一化，把这个矩阵也加入列表，[3*3,3*1]
        self.W.append(w2 / np.sqrt(layers[-2]))
        # print("W with bias trick:\n", len(self.W))

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

    # 定义tanh函数和它的导数
    def tanh(self,x):
        return 5 * np.tanh(x)

    def tanh_derivate(self,x):
        return 5 - 5 * np.tanh(x) * np.tanh(x)  # tanh函数的导数

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
        # out是每一层输出
        for layer in np.arange(0, len(self.W)-1): #２
            net = A[layer].dot(self.W[layer])
            out = self.tanh(net)
            A.append(out)

        for layer in np.arange(len(self.W)-1, len(self.W)): #２
            net = A[layer].dot(self.W[layer])
            out = self.tanh(net)
            A.append(out)

        #print(A[-1])
        #print(y)

        # 反向传播(Back Propagation, BP)
        error = A[-1] - y# 最后一层的输出减去实际结果
        #print(error)
        D = [error * self.tanh_derivate(A[-1])]
        # 反向
        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.tanh_derivate(A[layer])
            D.append(delta)

        D = D[::-1] # 倒序
        # 更新权值W
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    # 预测
    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
        # print(p)
        for layer in np.arange(0, len(self.W)-1):
            # print(self.W[0])
            p = self.tanh(np.dot(p, self.W[layer])) # np.dot矩阵乘法，点乘
        for layer in np.arange(len(self.W)-1, len(self.W) ):
            # print(self.W[0])
            p = self.tanh(np.dot(p, self.W[layer]))  # np.dot矩阵乘法，点乘

        return p

    # 计算loss，就是计算均方误差
    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)/100

        return loss


if __name__ == '__main__':
    nn = NeuralNetwork([2, 2, 1])
    print(nn)