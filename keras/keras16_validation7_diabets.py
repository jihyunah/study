# [괒[, 실습]
# R2 0.62 이상

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.8, shuffle=True, random_state=72
)

print(x)
print(x.shape)  # (442, 10)
print(y)
print(y.shape)  # (442, )

print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(250))
model.add(Dense(320))
model.add(Dense(500))
model.add(Dense(330))
model.add(Dense(210))
model.add(Dense(130))
model.add(Dense(100))
model.add(Dense(56))
model.add(Dense(30))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
model.fit(x_train, y_train, epochs=1600, batch_size=10, 
          validation_split=0.1)

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

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)

'''
발리데이션 안쓰는 게 더 높게 나옴
RMSE:  46.9743224034423
R2:  0.6658316010244898

발리데이션=0.1로는 
RMSE:  47.3411158069893
R2:  0.6605925981934805

'''