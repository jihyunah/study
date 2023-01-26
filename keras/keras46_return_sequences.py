import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM



#1. 데이터 
x = np.array([[1,2,3],[2,3,4],[3,4,5], 
              [4,5,6],[5,6,7], [6,7,8], 
              [7,8,9], [8,9,10], [9,10,11], 
              [10,11,12],[20,30,40], 
              [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])   #난 원해 80을 !


print(x.shape, y.shape) # (13, 3) (13,)
x = x.reshape(13, 3, 1)

#2. 모델구성

model = Sequential()
model.add(LSTM(128, input_shape=(3,1), return_sequences=True,  # return_squences=True를 써서 1번째 차원의 차원을 고대로 가져가줌. 
            #    --> LSTM 두번 쓰고 싶으면 return_sequences=True 사용하면 됨.
               activation='relu')) # 아웃풋 (N, 64) --> LSTM은 3차원을 2차원으로 아웃풋을 내놓음. 
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()
'''
#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss:', loss)
x_predict = np.array([50, 70, 60]).reshape(1, 3, 1) # [50, 60, 70]는 (3,1)임. 
result = model.predict(x_predict)
print('[50,60,70]의 결과 :', result)
'''
# loss: 0.006236345041543245
# [50,60,70]의 결과 : [[80.27065]]