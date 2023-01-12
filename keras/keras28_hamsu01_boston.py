from tensorflow.keras.models import Sequential, Model #모델이 함수형
from tensorflow.keras.layers import Dense, Input  #함수형에는 인풋 레이어 해야함. 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1.데이터
dataset = load_boston()
x = dataset.data
y=  dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=44                                             
) 

scaler = MinMaxScaler()
# scaler = StandardScaler()

scaler.fit(x_train) # --> 가중치를 생성시킨다는 것
x_train = scaler.transform(x_train) # 위에서 한 가중치 값을 transfrom으로 자 너 이제 시작 하는 것임
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 
# 스케일은 무조건 train만 한다. test, validation은 train의 값에 맞에 맞춰 스케일한다. 




print (x) 
print (x.shape) #(506, 13)
print (y)       
print (y.shape) #(506, )

print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']
print(dataset.DESCR) 

#2. 모델구성 (순차형)

# model = Sequential()
# model.add(Dense(50, input_dim=13, activation='relu'))
# model.add(Dense(40, activation='sigmoid'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='linear'))
# model.add(Dense(1, activation='linear'))
# model.summary()
# Total params: 4,611

#2. 모델구성 (함수형)
# 정의를 먼저 하는게 아니라, 모델 구성 먼저하고, 마지막에 이게 함수형 모델이야. 라고 정의해줌. 

input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1)
# model.summary()
# Total params: 4,611

'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
model.fit(x_train, y_train, epochs=10000, batch_size=32, 
          validation_split=0.25,verbose=1)


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

'''
'''
scaler 변환 전 
RMSE :  4.412067269906501
R2 :  0.7759482750052653

minmax scaler 변환 후 
RMSE :  2.7828259222525817
R2 :  0.9108674930959545

standard scaler 변환 후
RMSE :  2.851432881515548
R2 :  0.9064184254975071
'''

'''

#mse mae 아무거나 사용 adam 사용 metrix는 넣어도 되고 안 넣어도 됨
# model.fit에 x트레인 y트레인 데이터 들어감 batch size(디폴트 32)와 epochs 조절
# evaluate에 train test split 에서 분리한 x 와 y 들어감
#  R2 rmse 지표에 넣어줄 건 x test를 통한 y predict를 x test와 비교

'''
'''
1. random_state=44
   모델 100-90-80-80...
   epochs=10000, batch_size=32
   --> r2: 0.756...
'''