import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input

x1_datasets = np.array([range(100), range(301, 401)]).transpose()
print(x1_datasets.shape)   # (100, 2)  # 삼성전자의 시가, 고가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).T #transfpos 와 같음
print(x2_datasets.shape)   # (100, 3)   # 아모레의 시가, 고가, 종가

y = np.array(range(2001, 2101)) #(100, )    # 삼성전자의 월요일 시가   # 컬럼 한개인 것임

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y, train_size=0.7, random_state=1234
)    # train_test_split 3개 이상도 넣을 수 있음 

print(x1_train.shape, x2_train.shape, y_train.shape) # (70, 2) (70, 3) (70,)
print(x2_test.shape, x2_test.shape, y_test.shape) # (30, 3) (30, 3) (30,) # 잘 train,test 잘린거 알 수 있음 

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

#2-3. 모델병합
from tensorflow.keras.layers import concatenate  # concatenate 사슬같이 잇다; 연쇄시키다;
merge1 = concatenate([output1, output2], name='mg1') # 2개 이상은 리스트 형태
merge2 = Dense(12, activation='relu', name='mg2')(merge1)
merge3 = Dense(13, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)  # 모델이 두개라서 인풋이 2개임. 

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=10, batch_size=8)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss:', loss)


# 날짜를 순서대로 변환. 
# 여러개의 컬럼중에 5개 임의로 고를 것. 
# 앙상블 소스와 가중치를 제출 --> 가중치는 모델체크포인트로 파일로 해서 내기.   --> 삼성 월요일 시가를 맞추기. 
# 메일 제목 : 이지현 78,700원
# 1등 ~ 5등까지 시상. 