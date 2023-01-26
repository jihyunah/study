# https://www.kaggle.com/cp,petitions/bike-sharing-demand

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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
                                                    , random_state=333)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train) # --> 가중치를 생성시킨다는 것
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv)

print(x_train.shape, x_test.shape) # (7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) # (7620,) (3266,)
print(test_csv.shape)

x_train = x_train.reshape(7620, 2, 2, 2)
x_test = x_test.reshape(3266, 2, 2, 2)
test_csv = test_csv.reshape(6493, 2, 2, 2)


#2. 모델구성 (순차형)

model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(2, 2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(Conv2D(32, (2,2), input_shape=(2, 2, 2), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary() # Total params: 112,001







# activation 종류: relu , linear, sigmoid
# relu : 음수값들 전체를 0으로 만들고, 양수는 그대로 나온다. 
# 보통 마지막 레이어에는 사용하지 않고, 중간 레이어에 사용함.
# linear : activation의 기본 디폴트값. 선을 그어주는 것임
# sigmoid : 값을 0~1 사이로 만들어줌. 0.5 이상이면 1로 표현. 마지막 레이어에 쓰지 않음.


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, 
                              verbose=1)

import datetime 
date = datetime.datetime.now()
date = date.strftime('%M%D_%H%M')

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1, 
                      filepath= filepath + 'k31_05_' + date + '_' + filename)

hist = model.fit(x_train, y_train, epochs=215, batch_size=100,
          validation_split=0.2, callbacks=[earlyStopping, mcp])

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE = RMSE(y_test, y_predict)

print('RMSE:', RMSE)

r2 = r2_score(y_test, y_predict)
print('R2:', r2)


# 제출할 놈

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) #(6493, 1)

submission['count'] = y_submit

submission.to_csv(path + 'submission_0109.csv')

print("===================================")
print(hist) # <keras.callbacks.History object at 0x000002593F1E46A0>
print("===================================")
print(hist.history['loss']) #loss 값을 리스트로 정리해놓음. ''으로 되어있는건 키, []로 리스트 형태로 묶인건 밸류. 이 전체의 형태를 딕셔너리 형태. 
#그래서 이 딕셔너리는 키와 밸류로 구성되어 있다. 리스트는 두개 이상의 형태. 
print("===================================")
print(hist.history['val_loss']) 

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6)) #그림판의 사이즈 정하는 것
plt.plot(hist.history['loss'], c='red', marker='.', label='loss') #plot은 선 긋는 것, x가 어차피 에포 순대로 들어가기 때문에 y만 적어주면 됨. 
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss') #c는 컬러, marker는 선의 형태, 라벨은 선의 이름
plt.grid() # 격자, 모눈종이 형태로 보일 것임
plt.xlabel('epochs') #x축의 이름 설정
plt.ylabel('loss')
plt.title('kaggle_bike loss')
plt.legend() # 선의 이름을 보여줌
# plt.legend(loc='upper left') # loc는 location 위치의 줄임말
plt.show()


'''
RMSE: 149.56289238190675
R2: 0.3078589613749019

minmax scaler:
RMSE: 147.39942506638425
R2: 0.32773814621402975


standard scaler:
RMSE: 152.5407169631458
R2: 0.28002327630981305

drop out:
RMSE: 146.66491054022907
R2: 0.334421425878906

'''
# R2: 0.32618627088902075
