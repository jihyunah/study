import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

#1. 데이터           # 10일의 주식 데이터라고 가정
dataset = np.array([1,2,3,4,5,6,7,8,9,10]) # (10, )
# y는 없음 실질적으로.

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])  
# 3일치로 잘랐다. 

y = np.array([4, 5, 6, 7, 8, 9, 10])
# 4일치의 데이터 예측

print(x.shape, y.shape)  # (7, 3) (7,)

x = x.reshape(7, 3, 1)  # 1-->2-->3 순차적으로 연산하기 위해 1개씩 나눠서 reshape 잡아둔 것임. [[1],[2],[3]]
print(x.shape)

#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(128 , activation='relu', input_shape=(3, 1)))  # rnn 
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss:', loss)
y_pred = np.array([8, 9, 10]).reshape(1, 3, 1) # [8, 9, 10]는 (3,1)임. 
result = model.predict(y_pred)
print('[8,9,10]의 결과 :', result)

# loss: 4.1640350900706835e-06
# [8,9,10]의 결과 : [[11.00278]]