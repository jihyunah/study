# 1/30 월 수업 진도 시작~

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
x_datasets = np.array([range(100), range(301, 401)]).transpose()
print(x_datasets.shape)   # (100, 2)  # 삼성전자의 시가, 고가

y1 = np.array(range(2001, 2101)) #(100, )    # 삼성전자의 월요일 시가   # 컬럼 한개인 것임
y2 = np.array(range(201, 301))

x_train, x_test, y1_train, y1_test, \
    y2_train, y2_test = train_test_split(
    x_datasets, y1, y2, train_size=0.7, random_state=1234
)    # train_test_split 3개 이상도 넣을 수 있음 

print(x_train.shape, y1_train.shape, y2_train.shape) # (70, 2) (70, 3) , (70, 2), (70,)
print(x_test.shape, y1_test.shape, y2_test.shape) # (30, 3) (30, 3) , (30, 2), (30,) # 잘 train,test 잘린거 알 수 있음 

#2. 모델구성

#2-1. 모델 1.
input1= Input(shape=(2,))
dense1= Dense(11, activation='relu', name='ds11')(input1)
dense2= Dense(12, activation='relu', name='ds12')(dense1)
dense3= Dense(13, activation='relu', name='ds13')(dense2)
output1= Dense(14, activation='relu', name='ds14')(dense3)   # concatenate 의 input


#2-2. 분기모델 


#2-2. y1
dense40= Dense(11, activation='relu', name='ds40')(output1)
dense41= Dense(12, activation='relu', name='ds41')(dense40)
dense42= Dense(13, activation='relu', name='ds42')(dense41)
last_output1= Dense(14, activation='relu', name='lastoutput1')(dense42)

#2-3. y2
dense43= Dense(11, activation='relu', name='ds43')(output1)
dense44= Dense(12, activation='relu', name='ds44')(dense43)
dense45= Dense(13, activation='relu', name='ds45')(dense44)
last_output2= Dense(14, activation='relu', name='lastoutput2')(dense45)

model = Model(inputs=input1, outputs=[last_output1, last_output2])  # 모델이 두개라서 인풋이 2개임. 


model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', 
                   restore_best_weights=True, patience=50, verbose=1)

model.fit(x_train, [y1_train, y2_train],epochs=100, batch_size=1, 
          callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, [y1_test, y2_test])
print('loss:', loss)

# loss: 0.0010468135587871075



#  Y 모델이 2개면 로스는 2개의 로스와, 그 로스 두개를 합친 값 해서 총 3개의 로스값이 나온다. 