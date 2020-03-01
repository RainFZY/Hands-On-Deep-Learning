import gzip
import request
import numpy as np
import pickle
from six.moves import urllib
import random

filename = [
	["training_images","train-images-idx3-ubyte.gz"],
	["test_images","t10k-images-idx3-ubyte.gz"],
	["training_labels","train-labels-idx1-ubyte.gz"],
	["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        urllib.request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

def get_batch(X, Y, batch_size): # 64 batch_size
    N = len(X)
    i = random.randint(1, N-batch_size) # 产生两个数之间一个整数型随机数
    return X[i:i+batch_size], Y[i:i+batch_size]

load()
X_train, Y_train, X_test, Y_test = load()
X_train, X_test = X_train/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)


print(X_train.shape) # 数量，输入feature map 面积 # 60000, 784
print(len(Y_train))
print(X_test.shape[0])
print(len(Y_test))

test_size = 100
print("test_size = " + str(test_size))
X_train_min, Y_train_min = get_batch(X_train, Y_train, test_size)
X_test_min,  Y_test_min  = get_batch(X_test,  Y_test,  test_size)
print(len(X_train_min))
print(len(Y_train_min))
print(len(X_test_min))
print(len(Y_test_min))