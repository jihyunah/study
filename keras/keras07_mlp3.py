import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)])
# print(range(10)) : 0~9
print(x.shape) # (3,10)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
print(y.shape) #(2,10)

x = x.T
y = y.T

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(40))
model.add(Dense(32))
model.add(Dense(15))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=1250, batch_size=25)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss:', loss)

result = model.predict([[9, 30, 211]])
print('[[9,30,211]]의 예측값:', result)

'''
결과: [[10.663534   -0.32523203]]
로스값: 1.3721357583999634
'''