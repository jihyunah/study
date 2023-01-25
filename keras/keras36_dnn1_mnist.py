import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)      # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)      # (10000, 28, 28) (10000,)

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
# =============================데이터 전처리. 2차원으로 변환 ===============================
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape)      # (60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape)        # (10000, 28, 28, 1) (10000,)

print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.8, random_state=333)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dropout

#2. 모델구성
model = Sequential()                                # 28*28
model.add(Dense(128, activation='relu', input_shape=(784, )))             # (28, 28, 128) # 패딩은 겉 가장자리에 한겹 씌워주는 것. 그래서 사이즈를 동일하게 유지하게 해줌.
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu')) 
model.add(Dense(10, activation='softmax'))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=20, verbose=1)

import datetime 
date = datetime.datetime.now()
date = date.strftime('%M%D_%H%M')

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1, 
                      filepath= filepath + 'k34_1_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32, validation_data=[x_val, y_val], 
          callbacks=[es, mcp])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss:', results[0])
print('acc:', results[1])


# 얼리 스타핑, 모델체크포인트 적용/ val 적용

# padding, maxpool 안썼을 때 기본 성능 acc: 0.9803749918937683
# padding 적용시 acc: 0.9810000061988831
# Maxpool 적용시 acc: 0.9897500276565552
# CNN을 dnn으로 바꿔서 한 결과 : acc: 0.9721249938011169

