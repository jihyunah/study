import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(10))
# print(range(10)) : 0~9
print(x.shape) # (10,)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
              [9,8,7,6,5,4,3,2,1,0]])
print(y.shape) #(3,10)

x = x.T
y = y.T

# [실습] train_test_split를 이용하여
# 7:3으로 잘러서 모델 구현

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=123)


print('x_train: ', x_train)
print('x_test: ', x_test)
print('y_train: ', y_train)
print('y_test: ', y_test)
