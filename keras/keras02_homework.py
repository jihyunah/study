import tensorflow as tf

import numpy as np 

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련시키기
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. 평가, 예측
result = model.predict([13])
print('결과:', result)


# [13] : 을 예측해봐요