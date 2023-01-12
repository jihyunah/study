import numpy as np
from sklearn.datasets import fetch_covtype
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]), 
# array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))

#########################1. 케라스 투 카테고리컬 (원핫인코딩) ###############################
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y) #--> 이걸로 하면 안됨. 이거는 데이터에 0이 없어도 0을 자동으로 만들어서 8 컬럼으로 진행해야됨. --> np.delete로 삭제
# print(y.shape) # (581012, 8)
# print(type(y)) # <class 'numpy.ndarray'>
# print(y[:10])
# print(np.unique(y[:,0], return_counts=True)) #--> 모든 행의 0번째를 보여달라. = 0 칼럼을 보여달라
# print('======================================')
# y = np.delete(y, 0, axis=1)
# print(y.shape) # (581012, 7)
# print(y[:10])
# print(np.unique(y[:,0], return_counts=True)) #--> 0,1 이 나오니까 0칼럼 잘 지워진 것임. 

# y = np.delete(y, 0, axis=1)
# print(y.shape) # (581012, 7)
# print(y)

import tensorflow as tf

#########################2. 판다스 겟 더미스 (원핫인코딩) ###############################


# y = pd.get_dummies(y)
# print(y[:10])
# print(type(y)) # <class 'pandas.core.frame.DataFrame'>
# # --> 판다스에서는 인덱스와 헤더(컬럼)가 자동으로 생성된다. 
# # --> 현재 겟더미를 지났기 때문에 y가 판다스 형태인 것.
# # y = y.values # --> 판다스 형태의 y를 넘파이로 바꿈
# y = y.to_numpy() # --> values, to_numpy나 둘 중 하나 사용하면 됨
# print(type(y)) # <class 'numpy.ndarray'>
# print(y.shape) # (581012, 7)


#########################3. 사이킷런 원핫인코딩 (원핫인코딩) ###############################
print(y.shape) # (581012,) --> 1차원
y = y.reshape(581012, 1) # --> 2차원으로 바꿔줌
print(y.shape) # (581012, 1)
from sklearn.preprocessing import OneHotEncoder # --> preprocessing:전처리 라는 뜻
ohe = OneHotEncoder()
# ohe.fit(y) # y를 원핫인코딩 하겠다는 뜻. 
# y = ohe.transform(y)  --> 얘네를 한줄로 만들면 아래와 같음.
y = ohe.fit_transform(y)
y = y.toarray() # -> numpy 형태로 바꿔주는 것. 

print(y[:15])
print(type(y)) # <class 'scipy.sparse._csr.csr_matrix'>
print(y.shape) # (581012, 7)


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, 
                                                     train_size=0.8, stratify=y)
#2. 모델 구성
model = Sequential()
model.add(Dense(130, input_shape=(54,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor='accuracy', mode='max', patience=40, restore_best_weights=True, 
                              verbose=1)
hist = model.fit(x_train, y_train, epochs=2, batch_size=256, 
                 validation_split=0.2, callbacks=[earlyStopping])

#4. 평가, 예측
loss, accucary = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accucary:', accucary)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1) 
# --> 넘파이 자료형이 판다스 형태를 바로 못받아들인다. 
# 하지만 y_predict는 넘파이로 바뀜
print('y_predict(예측값):', y_predict[:20])

y_test = np.argmax(y_test, axis=1)
print('y_test(원래값):', y_test[:20])
# --> 넘파이 자료형이 판다스 형태를 바로 못받아들인다. 
# 그래서 y_test를 넘파이로 바꿔줘야 한다. 

acc = accuracy_score(y_test, y_predict)
print('accuracy score :', acc)

# loss: 0.27905645966529846
# accucary: 0.8936516046524048
# y_predict: [1 6 4 ... 1 1 0]
# y_test: [1 6 4 ... 1 1 0]
# accuracy score : 0.8936516268943142

