# 발리데이션, 액티베이션
# # [실습]
# R2 0.55~0.6 이상

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train) # --> 가중치를 생성시킨다는 것
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

print(x)
print(x_train.shape, x_test.shape)  # (14447, 8) (6193, 8)

x_train = x_train.reshape(14447, 2, 2, 2)
x_test = x_test.reshape(6193, 2, 2, 2)
print(x_train.shape, x_test.shape)  # (14447, 2, 2, 2) (6193, 2, 2, 2)



# print(datasets.feature_names)
# # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
# print(datasets.DESCR)

#2. 모델구성 (순차형)
model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(2, 2, 2), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary() # Total params: 252,916




#3 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, 
                              verbose=1)
import datetime 
date = datetime.datetime.now()
date = date.strftime('%M%D_%H%M')

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1, 
                      filepath= filepath + 'k31_02_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=2950, batch_size=100,
          validation_split=0.25, callbacks=[earlyStopping, mcp])

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
값 조정
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=66
)
model.fit(x_train, y_train, epochs=3200, batch_size=100)
RMSE:  0.6851476950858859
R2:  0.6578941993890843

--> hist로 보니까 2950번 에포가 좋은 것 같음. 

minmax scaler:
RMSE:  0.6295503373282532
R2:  0.7111629120345726


standard scaler:
RMSE:  0.5760116554682116
R2:  0.7582009564767547

dropout 안하는게 더 좋음. 
'''

# cnn R2:  0.7914411272227986