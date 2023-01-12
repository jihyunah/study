# [실습]
# 1. train 0.7 이상
# 2. R2: 0.8 이상/ RMSE 사용
# 평가지표: R2, RMSE

# validation, activation 사용

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston

#1.데이터
dataset = load_boston()
x = dataset.data
y=  dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
       train_size=0.7, shuffle=True, random_state=44                                             
) 



print (x) 
print (x.shape) #(506, 13)
print (y)       
print (y.shape) #(506, )

print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']
print(dataset.DESCR) 

#2. 모델구성

model = Sequential()
model.add(Dense(100, input_dim=13, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
model.fit(x_train, y_train, epochs=10000, batch_size=32, 
          validation_split=0.25)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

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
RMSE :  4.412067269906501
R2 :  0.7759482750052653
'''

#mse mae 아무거나 사용 adam 사용 metrix는 넣어도 되고 안 넣어도 됨
# model.fit에 x트레인 y트레인 데이터 들어감 batch size(디폴트 32)와 epochs 조절
# evaluate에 train test split 에서 분리한 x 와 y 들어감
#  R2 rmse 지표에 넣어줄 건 x test를 통한 y predict를 x test와 비교

'''
1. random_state=44
   모델 100-90-80-80...
   epochs=10000, batch_size=32
   --> r2: 0.756...
'''