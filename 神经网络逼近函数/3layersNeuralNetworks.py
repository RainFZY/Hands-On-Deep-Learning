import numpy as np
import matplotlib.pyplot as plt
import math

#设置迭代次数和学习率
epoch = 5000
learningRate = 0.5
#权重的行数自己设定，列数应该和输入的列数一致
w1_column =100

#生成数据集
X = np.linspace(-1.5, 1.5, 30)
'''X = np.array([[0,0],[0,1],[1,0],[1,1]])'''
#得到x的形状
xSize = X.shape
#有多少个x，就有多少个y
'''y = np.array([0,1,1,0])'''
y = np.zeros(xSize[0])
#为y赋值，这里按实际要逼近的函数来算
for i in range(xSize[0]):
    y[i] = 1/math.sin(X[i])+1/math.cos(X[i])
    #y[i] = math.sin(X[i])
if len(xSize)==1:
    w1_row = 1
else:
    w1_row = xSize[1]

#初始化权重和偏置量
#输入与sigmoid之间的偏置量和权重
w1 = np.random.random((w1_column, w1_row))
b1 = np.random.random((w1_column, 1))
#sigmoid与输出之间的偏置量和权重
w2 = np.random.random((1, w1_column))
b2 = np.random.random((1, 1))

#定义sigmoid函数
def sigmoid(x):
    return(1/(1+math.exp(-x)))

errors = []
times = 2
for t in range(epoch):
    errorThis = 0
    #学习率随迭代次数下降
    if (t == epoch*(times-1)/(times)):
        learningRate = learningRate/2
        times = times*2
    for i in range(xSize[0]):
        #激活函数前
        if (w1_row == 1):
            noSig = np.dot(w1,X[i])+b1
        else:
            noSig = np.dot(w1, X[i].reshape(X[i].shape[0], 1)) + b1
        #激活函数后
        AfterSig = np.zeros((w1_column, 1))
        for j in range(w1_column):
            AfterSig[j][0] = sigmoid(noSig[j])
        #预测结果
        predictY = np.dot(w2,AfterSig)+b2
        #计算误差
        error = predictY - y[i]
        errorThis += abs(error)
        # 梯度回传,更新权重和偏置量
        b2 = b2 - learningRate*error
        for j in range(w1_column):
            b1[j] = b1[j] - learningRate*error*w2[0][j]*AfterSig[j]*(1-AfterSig[j])
        for j in range(w1_column):
            for s in range(w1_row):
                if (w1_row == 1):
                    w1[j][s] = w1[j][s] - learningRate*error*w2[0][j]*AfterSig[j]*(1-AfterSig[j])*X[i]
                else:
                    w1[j][s] = w1[j][s] - learningRate * error * w2[0][j] * AfterSig[j] * (1 - AfterSig[j]) * X[i][s]
        for j in range(w1_column):
            w2[0][j] = w2[0][j] - learningRate*error*AfterSig[j]
    errors.append(errorThis[0][0])

Xtest = X
Ytest = np.zeros(xSize[0])
for i in range(xSize[0]):
    # 激活函数前
    if (w1_row == 1):
        noSigTest = np.dot(w1, np.transpose(Xtest[i])) + b1
    else:
        noSigTest = np.dot(w1, Xtest[i].reshape(Xtest[i].shape[0], 1)) + b1
    # 激活函数后
    AfterSigTest = np.zeros((w1_column, 1))
    for j in range(w1_column):
        AfterSigTest[j] = sigmoid(noSigTest[j])
    # 预测结果
        Ytest[i] = np.dot(w2,AfterSigTest) + b2


#画出损失函数的图，查看收敛情况
xaxis = range(len(errors))
plt.figure()
plt.plot(xaxis, errors, color='r', linewidth=3)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")


#画出原函数和拟合函数的图
xaxis = range(len(X))
plt.figure()
plt.plot(X, y, color='r', linewidth=3)
plt.plot(Xtest, Ytest,color='g', linewidth=3)
plt.title("Fitting Curve")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
