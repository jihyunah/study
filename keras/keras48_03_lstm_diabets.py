# [괒[, 실습]
# R2 0.62 이상

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.8, shuffle=True, random_state=72)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train) # --> 가중치를 생성시킨다는 것
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

# print(x)
print(x_train.shape, x_test.shape)  # (353, 10) (89, 10)

x_train = x_train.reshape(353, 5, 2)
x_test = x_test.reshape(89, 5, 2)



# print(datasets.feature_names)
# # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(datasets.DESCR)

#2. 모델구성 (순차형)
model = Sequential()
model.add(LSTM(64, input_shape=(5, 2), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary() # Total params: 554,597



model.summary() # Total params: 554,597


#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, 
                              verbose=1)

# import datetime 
# date = datetime.datetime.now()
# date = date.strftime('%M%D_%H%M')

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1, 
#                       filepath= filepath + 'k31_03_' + date + '_' + filename)

hist = model.fit(x_train, y_train, epochs=1600, batch_size=10, 
          validation_split=0.1, callbacks=[earlyStopping])

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
plt.title('diabets loss')
plt.legend() # 선의 이름을 보여줌
# plt.legend(loc='upper left') # loc는 location 위치의 줄임말
plt.show()


'''
RMSE:  46.67205269729491
R2:  0.6701183684761419

minmax scaler:
RMSE:  46.54832249234729
R2:  0.6718651191053755


standard scaler:
RMSE:  47.03396198303995
R2:  0.6649825280260004

dropout:
RMSE:  46.71531496507632
R2:  0.6695065230121442
'''

# CNN R2:  0.6811525048895701
# LSTM R2:  0.3104140729190351