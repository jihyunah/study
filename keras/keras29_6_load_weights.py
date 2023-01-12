from tensorflow.keras.models import Sequential, Model, load_model #모델이 함수형
from tensorflow.keras.layers import Dense, Input  #함수형에는 인풋 레이어 해야함. 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

path = './_save/'
# path = '../_save'
# path = 'c:/study/_save'

#1.데이터
dataset = load_boston()
x = dataset.data
y=  dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=44                                             
) 

scaler = MinMaxScaler()
scaler.fit(x_train) # --> 가중치를 생성시킨다는 것
x_train = scaler.transform(x_train) # 위에서 한 가중치 값을 transfrom으로 자 너 이제 시작 하는 것임
x_test = scaler.transform(x_test) 
# 스케일은 무조건 train만 한다. test, validation은 train의 값에 맞에 맞춰 스케일한다. 




#2. 모델구성 (함수형)
# 정의를 먼저 하는게 아니라, 모델 구성 먼저하고, 마지막에 이게 함수형 모델이야. 라고 정의해줌. 

input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()
# Total params: 4,611

# model.save_weights(path + 'keras29_5_save_weights1.h5')  # 모델을 저장한다.  --> 훈련을 안했기 때문에 뻥 가중치다. 
# model.load_weights(path + 'keras29_5_save_weights1.h5') # 모델 저장은 안되고 가중치만 저장된다. 따라서 얘를 쓰려면 모델과 컴파일이 정의 되어 있어야 한다. # 에러



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
# earlyStopping = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=20, verbose=1)
# model.fit(x_train, y_train, epochs=50, batch_size=32, 
#           validation_split=0.25,verbose=1, callbacks=[earlyStopping])

model.load_weights(path + 'keras29_5_save_weights2.h5')  # 모델을 저장한다. 이 위치에서 사용해야 되는 것이다. 



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


