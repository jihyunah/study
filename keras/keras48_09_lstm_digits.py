import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



#1. 데이터

datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,)
print(np.unique(y, return_counts=True)) # 고유 값의 개수 배열을 반환
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
# array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))



from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, 
                                                    train_size=0.8, stratify=y)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train) # --> 가중치를 생성시킨다는 것
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

print(x_train.shape, x_test.shape) # (1437, 64) (360, 64)
x_train = x_train.reshape(1437, 8, 8)
x_test = x_test.reshape(360, 8, 8)



#2. 모델구성 (순차형)
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(8, 8)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary() # Total params: 36,308



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor='accuracy', mode='max', patience=20, 
                              restore_best_weights=True, 
                              verbose=1)
# import datetime 
# date = datetime.datetime.now()
# date = date.strftime('%M%D_%H%M')

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1, 
#                       filepath= filepath + 'k31_09_' + date + '_' + filename)

hist = model.fit(x_train, y_train, epochs=100, batch_size=1, 
                 validation_split=0.2, callbacks=[earlyStopping])

#4. 평가, 예측
loss, accucary = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accucary:', accucary)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print('y_predict:', y_predict)

y_test = np.argmax(y_test, axis=1)
print('y_test:', y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy score :', acc)



# accuracy score : 0.9833333333333333
# dropout: accuracy score : 0.9583333333333334

# cnn accuracy score : 0.975
# LSTM accuracy score : 0.9333333333333333