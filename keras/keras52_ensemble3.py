# 1/30 월 수업 진도 시작~

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).transpose()
print(x1_datasets.shape)   # (100, 2)  # 삼성전자의 시가, 고가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).T #transfpos 와 같음
print(x2_datasets.shape)   # (100, 3)   # 아모레의 시가, 고가, 종가
x3_datasets = np.array([range(100, 200), range(1301, 1401)]).T 
print(x3_datasets.shape)   # (100, 2)



y1 = np.array(range(2001, 2101)) #(100, )    # 삼성전자의 월요일 시가   # 컬럼 한개인 것임
y2 = np.array(range(201, 301))

x1_train, x1_test, x2_train, x2_test, \
    y1_train, y1_test = train_test_split(
    x1_datasets, x2_datasets, y1, train_size=0.7, random_state=1234
)    # train_test_split 3개 이상도 넣을 수 있음 

x3_train, x3_test, y2_train, y2_test = train_test_split(x3_datasets, y2, train_size=0.7, random_state=1234)

print(x1_train.shape, x2_train.shape, x3_train.shape, y1_train.shape, y2_train.shape) # (70, 2) (70, 3) , (70, 2), (70,)
print(x2_test.shape, x2_test.shape, x3_test.shape, y1_test.shape, y2_test.shape) # (30, 3) (30, 3) , (30, 2), (30,) # 잘 train,test 잘린거 알 수 있음 

#2. 모델구성

#2-1. 모델 1.
input1= Input(shape=(2,))
dense1= Dense(11, activation='relu', name='ds11')(input1)
dense2= Dense(12, activation='relu', name='ds12')(dense1)
dense3= Dense(13, activation='relu', name='ds13')(dense2)
output1= Dense(14, activation='relu', name='ds14')(dense3)   # concatenate 의 input

# 머지 할 것이기 때문에 아직 output 내지 않음. 

#2-2. 모델 2.
input2= Input(shape=(3,))
dense21= Dense(21, activation='linear', name='ds21')(input2)
dense22= Dense(22, activation='linear', name='ds22')(dense21)
output2= Dense(23, activation='linear', name='ds23')(dense22)  # concatenate 의 input


# 2-3 모델 3. 
input3= Input(shape=(2,))
dense11= Dense(11, activation='relu', name='ds110')(input3)
dense20= Dense(12, activation='relu', name='ds120')(dense11)
dense30= Dense(13, activation='relu', name='ds130')(dense20)
output3= Dense(14, activation='relu', name='ds140')(dense30)


#2-4. 모델병합
from tensorflow.keras.layers import concatenate, Concatenate  # concatenate 사슬같이 잇다; 연쇄시키다;
#                              소문자 concatenate는 클래스, Concatenate는 함수이다.           
# con = Concatenate()([output1, output2, output3])

# model1 = Dense(5)(con)
# model2 = Dense(5)(model1)
# lastoutput = Dense(1)(model2)


merge1 = Concatenate()([output1, output2, output3]) # 2개 이상은 리스트 형태
merge2 = Dense(12)(merge1)
merge3 = Dense(13)(merge2)
con_output = Dense(1)(merge3)


# model = Model(inputs=[input1, input2, input3], outputs=lastoutput)


#2-5. 모델5, 분기 1
dense40= Dense(11, activation='relu', name='ds40')(con_output)
dense41= Dense(12, activation='relu', name='ds41')(dense40)
dense42= Dense(13, activation='relu', name='ds42')(dense41)
last_output1= Dense(14, activation='relu', name='lastoutput1')(dense42)

#2-6. 모델6, 분기 2
dense43= Dense(11, activation='relu', name='ds43')(con_output)
dense44= Dense(12, activation='relu', name='ds44')(dense43)
dense45= Dense(13, activation='relu', name='ds45')(dense44)
last_output2= Dense(14, activation='relu', name='lastoutput2')(dense45)

model = Model(inputs=[input1, input2, input3], outputs=[last_output1, last_output2])


# print(con_output.shape)

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])  # 메트릭스 mae 하면 loss값으로 y의 수와 같은 값이 더 나온다. 
#                                  그리고 mse의 loss 값은 y의 개수와 동일한 값과 그 값들의 합이 나옴. --> \
    #                                                                                   총 y의 개수 + 1 개의 로스가 나온다. 

es = EarlyStopping(monitor='val_loss', mode='min', 
                   restore_best_weights=True, patience=50, verbose=1)

model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train],epochs=100, batch_size=1, 
          callbacks=[es])

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print('loss:', loss)

# loss: 0.0010468135587871075