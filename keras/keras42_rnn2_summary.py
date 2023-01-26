import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

#1. 데이터           # 10일의 주식 데이터라고 가정
dataset = np.array([1,2,3,4,5,6,7,8,9,10]) # (10, )
# y는 없음 실질적으로.

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])  
# 3일치로 잘랐다. 

y = np.array([4, 5, 6, 7, 8, 9, 10])
# 4일치의 데이터 예측

print(x.shape, y.shape)  # (7, 3) (7,)

x = x.reshape(7, 3, 1)  # 1-->2-->3 순차적으로 연산하기 위해 1개씩 나눠서 reshape 잡아둔 것임. [[1],[2],[3]]
print(x.shape)

#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(units = 128 , activation='relu', input_shape=(3, 1)))  # rnn # 아웃풋 노드의 개수 128을 unit이라고 부름. 
                                                        # (N, 3, 1) --> ([batch=데이터의 개수, timesteps(주가, 유가임. y가 없다), feature])
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))



model.summary()


# Total params = recurrent_weights + input_weights + biases
# = (num_units*num_units)+(num_features*num_units) + (1*num_units)
# = (num_features + num_units)* num_units + num_units
# 결과적으로
# ( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 + unit 개수 ) + ( 1 * unit 개수)
# 를 참조하여 위의 total params 를 구하면

# 선생님이 알려주신 내용
# units * (units + feature + bias) = parms
# 128 * (128 + 1+ 1) = 16640 --> dnn보다 연산량이 많다. 