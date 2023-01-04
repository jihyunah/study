#[실습]
#1. train 0.7이상
#2. R2 : 0.8 이상 / RMSE 사용

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np   
from sklearn.model_selection import train_test_split

import sklearn as sk
print(sk.__version__)  #1.1.3

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape) # (506,13)

print(x)
print(y)
print(y.shape) # (506,)

print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT']

print(dataset.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=0)

print(x_train.shape)


#2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=13))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(350))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

print('==============')
print(y_test)
print(y_predict)
print('===============') 


from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('RMSE:', RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print('R2:', r2)

# 평가지표: R2, RMSE
'''
RMSE: 5.607126832297183
R2: 0.6224141968469803
'''
