import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10]) # (10, )
y = np.array(range(10)) # (10, )

#실습: 넘파이 리스트 슬라이싱! 7:3으로 잘라라
x_train = x[:-3]
x_test = x[-3:]
y_train = y[0:7]
y_test = y[7:10]

print(x_train)
print(x_test)
print(y_train)
print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(28))
model.add(Dense(450))
model.add(Dense(65))
model.add(Dense(21))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=250, batch_size=25)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

result = model.predict([11])
print('11의 결과:', result)

'''
결과: [[9.678058]]
로스: 0.2614215314388275
'''



