import numpy as np   
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0) # ./_data/ddarung/train.csv
test_csv = pd.read_csv(path + 'test.csv', index_col=0) # 0번째 컬럼은 index야 데이터 아니야! 라고 명시하는 것임
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape) # (1459, 10) --> 카운트를 분리해준다면 (1459, 9)가 될 것이다.

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],

print(train_csv.info())
# Non_Null = 결측치 2개있는 거 있음. 결측치가 상당히 많다는 것은 데이터를 수집을 못했다는 것.
# 이 데이터 없는 놈들 어떻게 해결?-->결측치 있는 것들은 아예 삭제시키기 (이 방법이 무조건 좋다는 것은 아님)

print(test_csv.info())
#카운트는 없다. 

print(train_csv.describe())
# 정보를 볼 수 있다. 

x = train_csv.drop(['count'], axis=1)
print(x)  # (1459 rows x 9 columns)
y = train_csv['count']
print(y)
print(y.shape) # (1459,)


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.7, random_state=1234)

print(x_train.shape, x_test.shape) # (1021, 9) (438, 9)
print(y_train.shape, y_test.shape) # (1021,) (438,)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=9))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(70))
model.add(Dense(70))
model.add(Dense(70))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
print(y_predict)

# 결측치 나쁜놈!!
# 결측치 때문에 TO BE CONTINUE

'''
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE = RMSE(y_test, y_predict)
print('RMSE: ', RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print('R2:', r2)


# 제출할 놈
y_submit = model.predict(test_csv)
'''

