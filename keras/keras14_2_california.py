# [실습]
# R2 0.55~0.6 이상

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=66
)

print(x)
print(x.shape)  # (20640, 8)
print(y)
print(y.shape)  # (20640, )

print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(40))
model.add(Dense(110))
model.add(Dense(180))
model.add(Dense(275))
model.add(Dense(320))
model.add(Dense(160))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(65))
model.add(Dense(30))
model.add(Dense(1))

#3 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
start = time.time()
model.fit(x_train, y_train, epochs=3200, batch_size=100)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

print("============================")
print(y_test)
print(y_predict)
print("============================")

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))

print("걸린시간:", end - start)

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)


'''
값 조정
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=66
)
model.fit(x_train, y_train, epochs=3200, batch_size=100)
RMSE:  0.7624036727447604
R2:  0.5763941048091229


# epochs=100
cpu 걸린시간: 30초
gpu 걸린시간: 85초


'''