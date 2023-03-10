import numpy as np 
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, shuffle=True, 
                                                train_size=0.8, random_state=333)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten


#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=129, kernel_size=(2,2), input_shape=(28, 28, 1), 
                 activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode = 'min', restore_best_weights=True, 
                   patience=10, verbose=3)

import datetime
date = datetime.datetime.now()
date = date.strftime('%M%D_%H%M')

filepath = './_save/MCP/'
filename = '{epochs:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=3, 
                      filepath = filepath + 'k34_01' + date + '_' + filename)

model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=3, 
          validatio_data=(x_val, y_val), callbacks=[es, mcp])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss:', results[0])
print('acc:', results[1])


