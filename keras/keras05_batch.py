import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))  #(Dense(3, inut_dim=1))은 하나의 레이어인 것임
model.add(Dense(50))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(20))
model.add(Dense(1))  #어차피 add기에 input dim 첫번째꺼 빼고 안넣어도 됨 / 사이의 히든 레이어의 아웃풋 숫자는 많이 해도 되는 것임



#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=10, batch_size=2)

#4. 평가. 예측
result = model.predict([6])
print('6의 결과:', result)

