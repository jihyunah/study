import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten
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


# ====피쳐를 2로 바꿀거야=====
x_train = x_train.reshape(72, 2, 2, 1)
x_test = x_test.reshape(24, 2, 2, 1)
x_predict = x_predict.reshape(7, 2, 2, 1)




#2. 모델구성
model = Sequential()
model.add(Conv2D(128, (2,2), padding='same', input_shape=(2, 2, 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
model.add(Conv2D(16, (2,2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

#3. 훈련 컴파일
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1)

#4.  평가 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)
result = model.predict(x_predict)
print('예측 결과 :', result)

# loss: 0.011954396031796932
# 예측 결과 : [[ 99.83881 ]
#  [100.837425]
#  [101.83604 ]
#  [102.83464 ]
#  [103.83323 ]
#  [104.83183 ]
#  [105.830444]]
