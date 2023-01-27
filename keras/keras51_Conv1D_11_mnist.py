# keras 35_1 복사했음

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)      # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)      # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

print(x_train.shape, y_train.shape)      # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)        # (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.8, random_state=333)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling2D, Input, LSTM


#2. 모델구성
model = Sequential()
model.add(Conv1D(128, 2, padding='same', input_shape=(28, 28), 
                 activation='relu'))             # (28, 28, 128) # 패딩은 겉 가장자리에 한겹 씌워주는 것. 그래서 사이즈를 동일하게 유지하게 해줌.
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(Conv1D(16, 2, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(12, activation='relu')) # input_shape = (40000,)
model.add(Dense(10, activation='softmax'))

# input1 = Input(shape=(28, 28, 1))
# dense1 = Conv2D(filters=256, kernel_size=(2,2), padding='same', activation='relu')(input1)
# dense2 = MaxPooling2D()(dense1)
# dense3 = Conv2D(filters=128, kernel_size=(2,2), padding='same', activation='relu')(dense2)
# dense4 = MaxPooling2D()(dense3)
# dense5 = Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu')(dense4)
# dense6 = MaxPooling2D()(dense5)
# dense7 = Flatten()(dense6)
# dense8 = Dense(32, activation='relu')(dense7)
# output1 = Dense(10, activation='softmax')(dense8)
# model = Model(inputs=input1, outputs=output1)


model.summary()


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=10, verbose=1)

# import datetime 
# date = datetime.datetime.now()
# date = date.strftime('%M%D_%H%M')

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1, 
#                       filepath= filepath + 'k34_1_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32, validation_data=[x_val, y_val], 
          callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss:', results[0])
print('acc:', results[1])




# padding, maxpool 안썼을 때 기본 성능 acc: 0.9803749918937683
# padding 적용시 acc: 0.9810000061988831
# Maxpool 적용시 acc: 0.9897500276565552

# LSTM acc: 0.11450000107
# conv1d acc: 0.9816250205039978