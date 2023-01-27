# 49_2 복붙

import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, GRU, Conv1D, Flatten
from sklearn.model_selection import train_test_split

#1. 데이터
a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))  # 예상 y = 100~106

timesteps1 = 5  # x는 4개, y는 1개 
timesteps2 = 4

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):   # --> range만큼이 행이 된다. 열은 timesteps
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps1)

print(bbb)
print(bbb.shape)  # (96, 5)


x = bbb[:, :-1]
y = bbb[:, -1]

print(x.shape, y.shape)  # (96, 4) (96,)
# x = x.reshape(96, 4, 1)


# ========================predict는 y값 필요 없어서 x와 y로 나눌 필요 없기 때문에 
            #   타임스텝스를 아예 4로 줘버린다. 그럼 행은 range와 같으니까 10-4+1=7이 되고, (7,4,1)이 된다. ====

x_predict = split_x(x_predict, timesteps2)
print(x_predict)
print(x_predict.shape) # (7, 4)

# x_predict = x_predict.reshape(7, 4, 1)


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1234)
# ==========Train size의 디폴트값은 0.75이다. =============


print(x_train.shape, y_train.shape)  # (72, 4) (72,)
print(x_test.shape, y_test.shape)    # (24, 4) (24,)


x_train = x_train.reshape(72, 4, 1)
x_test = x_test.reshape(24, 4, 1)
x_predict = x_predict.reshape(7, 4, 1)




#2. 모델구성
model = Sequential()
# model.add(LSTM(128, input_shape=(4, 1), activation='relu'))
# model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu'), input_shape=(4, 1)))
#         # bidirectional은 모델이 아닌 그냥 양방향을 하겠다는 뜻으로 단독으로 사용 불가  
#             # bidirectional 0 = 133120 parms. x = 66560 parms
# model.add(GRU(64, activation='relu'))
model.add(Conv1D(128, padding='same', kernel_size=2, input_shape=(4, 1))) # 1D에서는 kernel_size를 숫자 하나만 적기. 어차피 1차원 선이니까.  
model.add(Conv1D(64, padding='same', kernel_size=2)) 
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()

#3. 훈련 컴파일
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1)

#4.  평가 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)
result = model.predict(x_predict)
print('예측 결과 :', result)

# loss: 0.000142033226438798
# 예측 결과 : [[100.0061  ]
#  [101.00585 ]
#  [102.005554]
#  [103.005226]
#  [104.00488 ]
#  [105.00451 ]
#  [106.00411 ]]
