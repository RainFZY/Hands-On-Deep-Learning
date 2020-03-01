from nn2 import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import math

X = np.zeros((100,1))
# x范围取0.3到1.3
for i in range(100):
    X[i] = 0.3 + i/100
    # X[i] = -3 + 0.06 * i
    # X[i] = 0.5 * math.pi * (i+1)/101

X_size = X.size
y = np.zeros((X_size,1))

# 函数表达式
for i in range(X_size):
    # y[i]= 1/(math.sin(X[i])+0.05) + 1/(math.cos(X[i])+0.05)
    y[i] = 1/math.sin(X[i]) + 1/math.cos(X[i])
    # y[i] = math.sin(X[i])

# 构造1-50-1结构的神经网络，可以改隐藏层节点数
nn2 = NeuralNetwork([1,50,1], alpha=0.01)
# 训练模型，更新得到最终不断迭代更新的weight矩阵
losses = nn2.fit(X, y, epochs=5000, step=1)

# 测试结果
yTest = np.zeros((X_size,1))
for i in range(X_size):
    yTest[i] = nn2.predict(X[i])

plt.style.use("ggplot")
plt.figure()
plt.title('Original Curve')
plt.plot(X, y)
plt.plot(X, yTest, color='green')

# 绘制loss曲线
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(losses)), losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

print("最终的weights：\n", nn2.W)