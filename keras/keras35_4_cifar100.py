from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)      # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)        # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.8, random_state=333)

# scaler = MinMaxScaler()
# x_train = np.reshape(1, -1)
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = np.reshape(50000, 32, 32, 3)

# 데이터 정규화(Data Regularization)
# 이 과정을 통해서 추후 학습할 신경망이 조금 더 학습이 원할히 될 수 있게함
x_train = x_train / 255
x_val = x_val / 255
x_test = x_test / 255



#2. 모델구성
# model = Sequential()
# model.add(Conv2D(filters=100, kernel_size=(2,2), input_shape=(32, 32, 3), 
#                  activation='relu'))             # (31, 31, 128)
# model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu')) # (30, 30, 64)
# model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu')) # (29, 29, 64) --> 
# model.add(Flatten()) # --> 
# model.add(Dense(32, activation='relu')) # input_shape = (40000,)
#                                         # (6만, 4만)이 인풋이다. 6만이 batch_size, 4만이 input_dim이다. 
# model.add(Dense(10, activation='softmax'))

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3), 
                 padding='same', strides=1))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', padding='same', 
                 strides=2))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))







#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=10, verbose=1)

import datetime 
date = datetime.datetime.now()
date = date.strftime('%M%D_%H%M')

filepath = '../_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1, 
                      filepath= filepath + 'k34_3_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=100, verbose=3, batch_size=32, validation_data=[x_val, y_val], 
          callbacks=[es, mcp])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss:', results[0])
print('acc:', results[1])

# acc: 0.5116249918937683