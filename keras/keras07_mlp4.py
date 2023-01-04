import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x = np.array(range(10))
# print(range(10)) : 0~9
print(x.shape) # (10,)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
              [9,8,7,6,5,4,3,2,1,0]])
print(y.shape) #(3,10)


y = y.T

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(40))
model.add(Dense(32))
model.add(Dense(15))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=1250, batch_size=25)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss:', loss)

result = model.predict([9])
print('[[9,30,211]]의 예측값:', result)

'''
결과: [[1.0078421e+01 1.6658069e+00 9.5676482e-03]]
로스값: 0.06534019112586975
'''