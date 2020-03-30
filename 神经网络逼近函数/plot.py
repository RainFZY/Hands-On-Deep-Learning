from nn import NeuralNetwork
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 生成的数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y_or = np.array([[0], [1], [1], [1]])
# y_and = np.array([[0], [0], [0], [1]])
y_xor = np.array([[0], [1], [1], [0]])
# 构造2-2-1结构的神经网络，2个节点输入层，2个节点的隐藏层，1个节点的输出层，可以改隐藏层节点数
nn = NeuralNetwork([2, 2, 1], alpha=0.5)
# 训练模型，更新得到最终不断迭代更新的weight矩阵
losses = nn.fit(X, y_xor, epochs=20000, step=100)

# 测试并输出结果
for (x, target) in zip(X, y_xor):
    pred = nn.predict(x)[0][0]
    result = 1 if pred > 0.5 else 0
    print("data={}, ground_truth={}, pred={:.4f}, result={}"
          .format(x, target[0], pred, result))

# 绘制初始点
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
cm_dark = mpl.colors.ListedColormap(['r', 'b'])
plt.scatter(X[:, 0], X[:, 1], marker="o", c=y_xor.ravel(), cmap=cm_dark, s=80)
# print(testY)

# 绘制loss曲线
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(losses)), losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

print("最终的weights：\n", nn.W)
