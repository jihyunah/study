# https://www.kaggle.com/cp,petitions/bike-sharing-demand

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
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

#2. 모델구성 (순차형)

# model = Sequential()
# model.add(Dense(256, input_dim=8, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='linear'))
# model.summary() # Total params: 112,001


#2. 모델구성 (함수형)

input1 = Input(shape=(8,))
dense1 = Dense(256, activation='relu')(input1)
dense2 = Dense(256, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(64, activation='relu')(dense3)
dense5 = Dense(32, activation='relu')(dense4)
dense6 = Dense(16, activation='relu')(dense5)
dense7 = Dense(8, activation='relu')(dense6)
output1 = Dense(1, activation='linear')(dense7)
model = Model(inputs=input1, outputs=output1)
model.summary() # Total params: 112,001



# activation 종류: relu , linear, sigmoid
# relu : 음수값들 전체를 0으로 만들고, 양수는 그대로 나온다. 
# 보통 마지막 레이어에는 사용하지 않고, 중간 레이어에 사용함.
# linear : activation의 기본 디폴트값. 선을 그어주는 것임
# sigmoid : 값을 0~1 사이로 만들어줌. 0.5 이상이면 1로 표현. 마지막 레이어에 쓰지 않음.


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True, 
                              verbose=1)
hist = model.fit(x_train, y_train, epochs=215, batch_size=100,
          validation_split=0.2, callbacks=[earlyStopping])

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

'''
# RMSE: 147.1805186121678
# R2: 0.32105710135602517
# random_state=1234, epochs=130, batch_size=32, 모델 256*2, 128,64,,
