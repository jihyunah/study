import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

a = np.array(range(1, 11)) # 1~10
timesteps = 5 # (하나에 5개씩 자르고싶어 라는 뜻)

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):   # --> range만큼이 행이 된다. 열은 timesteps
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)

print(bbb)
print(bbb.shape)

x = bbb[:, :-1] # 모든 행의 -1 열= 가장 끝열까지
y = bbb[:, -1] # 모든 행의 가장 끝열
x_predict = np.array([7,8,9,10])
print(x, y)
print(x.shape, y.shape) # (6, 4) (6,)

x = x.reshape(6, 4, 1)
x_predict = x_predict.reshape(1, 4, 1)

# 실습 
# LSTM 모델을 구성하기 

#2. 모델구성
model = Sequential()
model.add(LSTM(128, input_shape=(4, 1), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

#3. 훈련 컴파일
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

#4.  평가 예측
loss = model.evaluate(x, y)
print('loss:', loss)
result = model.predict(x_predict)
print('[7,8,9,10]의 결과 :', result)

# loss: 0.0010739980498328805
# [7,8,9,10]의 결과 : [[10.997002]]