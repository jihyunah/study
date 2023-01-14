import numpy as np 
import tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)

print(x_train[1000])
print(y_train[1000]) # 5

import matplotlib.pyplot as plt 
plt.inshow(x_train[1000], 'gray')
plt.show()
