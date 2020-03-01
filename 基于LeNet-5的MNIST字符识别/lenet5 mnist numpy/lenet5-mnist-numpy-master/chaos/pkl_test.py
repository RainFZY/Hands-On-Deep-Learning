import pickle
import numpy as np

# test_images, training_images, test_labels, training_labels
f = open('weights.pkl','rb')
data = pickle.load(f)

np.set_printoptions(threshold=np.inf) # 数据完全显示
print(data['training_images'][0].shape)
# print(data[9])