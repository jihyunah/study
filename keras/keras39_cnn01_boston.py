# 31_01 에서 복사해서 가져옴

from tensorflow.keras.models import Sequential, Model #모델이 함수형
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D  #함수형에는 인풋 레이어 해야함. 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1.데이터
dataset = load_boston()
x = dataset.data
y=  dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.8, shuffle=True, random_state=44                                             
) 

scaler = MinMaxScaler()
# scaler = StandardScaler()

scaler.fit(x_train) # --> 가중치를 생성시킨다는 것
x_train = scaler.transform(x_train) # 위에서 한 가중치 값을 transfrom으로 자 너 이제 시작 하는 것임
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 
# 스케일은 무조건 train만 한다. test, validation은 train의 값에 맞에 맞춰 스케일한다. 




# print (x) 
print (x_train.shape, x_test.shape) #(404, 13) (102, 13) --> 3차원으로 (13,1,1)로 바꿔줄 수 있음.
# print (y)       


#================cnn 모델에 적용하기 위해 4차원 데이터로 변환시켜줌===================#

x_train = x_train.reshape(404, 13, 1, 1)
x_test = x_test.reshape(102, 13, 1, 1)

print (x_train.shape, x_test.shape) # (404, 13, 1, 1) (102, 13, 1, 1) --> 4차원 데이터로 변환 완료



print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']
print(dataset.DESCR) 

#2. 모델구성 (순차형)

model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(13, 1, 1), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
# Total params: 4,611

#2. 모델구성 (함수형)
# 정의를 먼저 하는게 아니라, 모델 구성 먼저하고, 마지막에 이게 함수형 모델이야. 라고 정의해줌. 

# input1 = Input(shape=(13,))
# dense1 = Dense(50, activation='relu')(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(40, activation='sigmoid')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(30, activation='relu')(drop2)
# drop3 = Dropout(0.2)(dense3)
# dense4 = Dense(20, activation='linear')(drop3)
# output1 = Dense(1, activation='linear')(dense4)
# model = Model(inputs=input1, outputs=output1)
# model.summary()
# Total params: 4,611


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, 
                   restore_best_weights=True, verbose=1)
import datetime 
date = datetime.datetime.now()
date = date.strftime("%M%D_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1, 
                      filepath= filepath + 'k31_01_' + date + '_' + filename)


model.fit(x_train, y_train, epochs=500, batch_size=32, 
          validation_split=0.25,verbose=1, callbacks=[es, mcp])


#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)


y_predict = model.predict(x_test)

print("------------------")
print(y_test)
print(y_predict)
print("------------------")

def RMSE(y_test, y_predict):
         return np.sqrt(mean_squared_error(y_test, y_predict))
#mse에 루트를 씌운다 이것이 RMSE return은 출력


print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


# R2 :  0.8152120345235627