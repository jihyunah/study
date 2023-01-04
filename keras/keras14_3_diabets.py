# [과제]
# R2 0.62 이상

from sklearn.datasets import load_diabetes
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(x.shape) # (442, 10)
print(y)
print(y.shape) # (442,)
print(datasets.DESCR)
print(datasets.feature_names) # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)

#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=10))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=200, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)
print("==============")
print(y_test)
print(y_predict)
print("==============")

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print('R2:', r2)

'''

'''
