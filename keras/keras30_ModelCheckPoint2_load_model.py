from tensorflow.keras.models import Sequential, Model, load_model #모델이 함수형
from tensorflow.keras.layers import Dense, Input  #함수형에는 인풋 레이어 해야함. 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/'
# path = '../_save'
# path = 'c:/study/_save'

#1.데이터
dataset = load_boston()
x = dataset.data
y=  dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=44) 

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

print (x) 
print (x.shape) #(506, 13)
print (y)       
print (y.shape) #(506, )

print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']
print(dataset.DESCR) 


# #2. 모델구성 (함수형)

# input1 = Input(shape=(13,))
# dense1 = Dense(50, activation='relu')(input1)
# dense2 = Dense(40, activation='sigmoid')(dense1)
# dense3 = Dense(30, activation='relu')(dense2)
# dense4 = Dense(20, activation='linear')(dense3)
# output1 = Dense(1, activation='linear')(dense4)
# model = Model(inputs=input1, outputs=output1)
# model.summary()
# # Total params: 4,611


'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=20, verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath= path + 'MCP/keras30_ModelCheckPoint1.hdf5') # 가장 좋은 지점을 저장할거야



model.fit(x_train, y_train, epochs=5000, batch_size=32, 
          validation_split=0.2, verbose=1, callbacks=[es, mcp])
'''

model = load_model(path + 'MCP/keras30_ModelCheckPoint1.hdf5')


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


# MCP 저장: R2 :  0.8583521386000977
# load_model: R2 :  0.8583521386000977

