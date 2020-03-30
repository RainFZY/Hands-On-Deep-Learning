from nn2 import NeuralNetwork
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

class ApproachNetwork:
    def __init__(self, hidden_size=100, output_size=1):
        self.params = {'W1': np.random.random((1, hidden_size)),
                       'B1': np.zeros(hidden_size),
                       'W2': np.random.random((hidden_size, output_size)),
                       'B2': np.zeros(output_size)}

    @staticmethod
    def generate_data(fun, is_noise=True, axis=np.array([-1, 1, 100])):
        """
         产生数据集
        :param fun: 这个是你希望逼近的函数功能定义，在外面定义一个函数功能方法，把功能方法名传入即可
        :param is_noise: 是否需要加上噪点，True是加，False表示不加
        :param axis: 这个是产生数据的起点，终点，以及产生多少个数据
        :return: 返回数据的x, y
        """
        np.random.seed(0)
        x = np.linspace(axis[0], axis[1], axis[2])[:, np.newaxis]
        x_size = x.size
        y = np.zeros((x_size, 1))
        if is_noise:
            noise = np.random.normal(0, 0.1, x_size)
        else:
            noise = None

        for i in range(x_size):
            if is_noise:
                y[i] = fun(x[i]) + noise[i]
            else:
                y[i] = fun(x[i])

        return x, y

# 逼近函数 f(x)=sin(x)
def fun_sin(x0):
    return math.sin(x0)

x, y = network.generate_data(fun_sin, False, axis=np.array([-3, 3, 100]))
ax = plt.gca()
ax.set_title('data points')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.scatter(x, y)
plt.show()



@staticmethod
def sigmoid(x_):
    return 1 / (1 + np.exp(-x_))

def sigmoid_grad(self, x_):
    return (1.0 - self.sigmoid(x_)) * self.sigmoid(x_)

def predict(self, x_):
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['B1'], self.params['B2']

    a1 = np.dot(x_, W1) + b1
    z1 = self.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2

    return a2

def loss(self, x_, t):
    y_ = self.predict(x_)
    return y_, np.mean((t - y_) ** 2)

def gradient(self, x, t):
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['B1'], self.params['B2']
    grads = {}

    batch_num = x.shape[0]

    # forward
    a1 = np.dot(x, W1) + b1
    z1 = self.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2

    # backward
    dy = (a2 - t) / batch_num
    grads['W2'] = np.dot(z1.T, dy)
    grads['B2'] = np.sum(dy, axis=0)

    dz1 = np.dot(dy, W2.T)
    da1 = self.sigmoid_grad(a1) * dz1
    grads['W1'] = np.dot(x.T, da1)
    grads['B1'] = np.sum(da1, axis=0)

    return grads

# 根据上述计算的梯度，结合学习率，更新权重和偏置
for key in ('W1', 'B1', 'W2', 'B2'):
    self.params[key] -= self.learning_rate * grad[key]

def train_with_own(self, x_, y_, max_steps=100):
        for k in range(max_steps):
            grad = self.gradient(x_, y_)
            for key in ('W1', 'B1', 'W2', 'B2'):
                self.params[key] -= self.learning_rate * grad[key]
            pred, loss = network.loss(x_, y_)

            if k % 150 == 0:
                # 动态绘制结果图，你可以看到训练过程如何慢慢的拟合数据点
                plt.cla()
                plt.scatter(x, y)
                plt.plot(x, pred, 'r-', lw=5)
                plt.text(0.5, 0, 'Loss=%.4f' % abs(loss), fontdict={'size': 20, 'color': 'red'})
                plt.pause(0.1)

        # 关闭动态绘制模式
        plt.ioff()
        plt.show()





if __name__ == '__main__':
    network = ApproachNetwork()

    x, y = network.generate_data(network.fun_sin, False, axis=np.array([-3, 3, 100]))
    ax = plt.gca()
    ax.set_title('data points')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.scatter(x, y)
    plt.show()

    # 使用 自编代码 训练
    network.train_with_own(x, y, 3500)








