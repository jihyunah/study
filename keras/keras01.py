import tensorflow as tf 
# 텐서플로를 임포트(불러오다)합니다. 하지만 너무 길어서 as 뒤는 줄여준다. (주석을 다는 것임)
print(tf.__version__)

import numpy as np

#1. 데이터. array:배열하다
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
#Sequential은 순서대로 add 함수의 다양한 수를 넣겠다는 것이고,
#dense는 빽빽한 이라는 뜻으로 레이어를 완전연결층으로 나타내겠다는 것
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))
# 왜 아웃풋 딤이라고는 안쓰는가. dim은 차원이라는 것. 

#3. 컴파일, 훈련 (컴퓨터가 말을 알아듣게 한다는 것)
model.compile(loss='mae', optimizer='adam')
#optimizer = 최적화 함수라는 뜻
model.fit(x, y, epochs=1000)
#fit은 피트니스 할 때 그 핏이고, 훈련한다는 말임. epochs는 몇번 테스트 할 것인지. 

#4. 평가, 예측
result = model.predict([4])
print('결과 :', result)

#