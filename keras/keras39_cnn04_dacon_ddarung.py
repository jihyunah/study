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
path = './_data/ddarung/'
# path = '../_data/ddarung/'
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

train_csv = train_csv.dropna() # 결측치 전체 삭제

x = train_csv.drop(['count'], axis=1)
print(x)  # (1459 rows x 9 columns)
y = train_csv['count']
print(y)
print(y.shape) # (1459,)

#### 결측치 처리 1. 제거 ####
print(train_csv.isnull().sum()) #train파일의 결측치 숫자 보여주는것
print(train_csv.isnull().sum())
print(train_csv.shape)   #(1328,10)
print(submission.shape) # (715,1) --> 715개의 답을 달아야 하기때문에 여기서 결측치는 삭제하면 안된다. 





x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.7, random_state=123)

print(x_train.shape, x_test.shape) # (929, 9) (399, 9)
print(y_train.shape, y_test.shape) # (1021,) (438,)




scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train) # --> 가중치를 생성시킨다는 것
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv)

print(test_csv.shape)

x_train = x_train.reshape(929, 3, 3, 1)
x_test = x_test.reshape(399, 3, 3, 1)
test_csv = test_csv.reshape(715, 3, 3, 1)  # 우리가 바꾼 모델에 맞춰서 하려면 쉐이프를 바꿔줘야 함. test x와 같은 것임. 

#2. 모델구성 (순차형)
model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(3, 3, 1), padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary() # Total params: 792,847





#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, 
                              verbose=1)

import datetime 
date = datetime.datetime.now()
date = date.strftime('%M%D_%H%M')

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1, 
                      filepath= filepath + 'k31_04_' + date + '_' + filename)

hist = model.fit(x_train, y_train, epochs=3900, batch_size=32,
          validation_split=0.25, callbacks=[earlyStopping, mcp])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
print(y_predict)

# 결측치 나쁜놈!!
# 결측치 때문에 TO BE CONTINUE


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE = RMSE(y_test, y_predict)
print('RMSE:', RMSE)

r2 = r2_score(y_test, y_predict)
print('R2:', r2)


# 제출할 놈
y_submit = model.predict(test_csv)
print(y_submit.shape) # (715,1)


# .to_csv()를 사용해서
# submission_0105.csv를 완성하시오!

submission['count'] = y_submit  # submission의 count 컬럼에 우리가 예측한 값을 넣는다는 것. 


submission.to_csv(path + 'submission_01091250.csv')

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
plt.title('ddarung loss')
plt.legend() # 선의 이름을 보여줌
# plt.legend(loc='upper left') # loc는 location 위치의 줄임말
plt.show()






'''
RMSE: 48.0353928473667
R2: 0.651623306484248


minmax scaler:
RMSE: 41.69737688105775
R2: 0.7374911984465802


standard scaler:
RMSE: 43.84555317214622
R2: 0.7097464724514065

dropout:
RMSE: 41.78707444247661
R2: 0.7363605889405485


'''

# CNN R2: 0.6587149928374831