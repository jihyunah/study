# https://www.kaggle.com/cp,petitions/bike-sharing-demand

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터

path = './_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0) #인덱스 컬럼 지정하지 않으면 자동으로 인덱스 생성됨
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(train_csv.info())

x = train_csv.drop(['casual','registered','count'], axis=1)
y = train_csv['count']
print(y.shape, x.shape) # (10886,) (10886, 8)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True
                                                    , random_state=1234)

print(x_train.shape, x_test.shape) # (7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) # (7620,) (3266,)

#2. 모델구성

model = Sequential()
model.add(Dense(256, input_dim=8, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# activation 종류: relu , linear, sigmoid
# relu : 음수값들 전체를 0으로 만들고, 양수는 그대로 나온다. 
# 보통 마지막 레이어에는 사용하지 않고, 중간 레이어에 사용함.
# linear : activation의 기본 디폴트값. 선을 그어주는 것임
# sigmoid : 값을 0~1 사이로 만들어줌. 0.5 이상이면 1로 표현. 마지막 레이어에 쓰지 않음.

#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
start = time.time()
model.fit(x_train, y_train, epochs=130, batch_size=100)
end = time.time()

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE = RMSE(y_test, y_predict)

print('RMSE:', RMSE)
print("걸린시간:", end - start)

r2 = r2_score(y_test, y_predict)
print('R2:', r2)


# 제출할 놈

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) #(6493, 1)

submission['count'] = y_submit
print(submission)

submission.to_csv(path + 'submission_0106.csv')

# RMSE: 147.1805186121678
# R2: 0.32105710135602517
# random_state=1234, epochs=130, batch_size=32, 모델 256*2, 128,64,,