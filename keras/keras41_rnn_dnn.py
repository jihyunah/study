import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터           # 10일의 주식 데이터라고 가정
dataset = np.array([1,2,3,4,5,6,7,8,9,10]) # (10, )
# y는 없음 실질적으로.

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])  
# 3일치로 잘랐다. 

y = np.array([4, 5, 6, 7, 8, 9, 10])
# 4일치의 데이터 예측

print(x.shape, y.shape)  # (7, 3) (7,)


#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(128 , activation='relu', input_shape=(3, 1)))  # rnn 
model.add(Dense(128, activation='relu', input_shape=(3, )))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()
#  512 = 128 * 3+1

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss:', loss)
y_pred = np.array([8, 9, 10]).reshape(3, 1) #.reshape(1, 3, 1) # [8, 9, 10]는 (3,1)임. 
result = model.predict(y_pred)
print('[8,9,10]의 결과 :', result)

# loss: 5.278317388203446e-13
# [8,9,10]의 결과 : [[11.072954]]