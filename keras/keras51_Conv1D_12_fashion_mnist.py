from tensorflow.keras.datasets import fashion_mnist 
import numpy as np
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling2D, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)      # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)      # (10000, 28, 28) (10000,)

print(x_train[1000])
print(y_train[1000])  # 5

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.8, random_state=333)

print(np.unique(y_train, return_counts=True))

# import matplotlib.pyplot as plt 
# plt.imshow(x_train[10], 'gray')
# plt.show()


#2. 모델구성
model = Sequential()
model.add(Conv1D(128, 2, padding='same', input_shape=(28, 28), 
                 activation='relu'))             # (27, 27, 128)
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(Conv1D(16, 2, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(12, activation='relu')) # input_shape = (40000,)
                                        # (6만, 4만)이 인풋이다. 6만이 batch_size, 4만이 input_dim이다. 
model.add(Dense(10, activation='softmax'))

# input1 = Input(shape=(28, 28, 1))
# dense1 = Conv2D(filters=128, kernel_size=(2,2), padding='same', activation='relu')(input1)
# dense2 = MaxPooling2D()(dense1)
# dense3 = Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu')(dense2)
# dense4 = MaxPooling2D()(dense3)
# dense5 = Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu')(dense4)
# dense6 = MaxPooling2D()(dense5)
# dense7 = Flatten()(dense6)
# dense8 = Dense(32, activation='relu')(dense7)
# output1 = Dense(10, activation='softmax')(dense8)
# model = Model(inputs=input1, outputs=output1)



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


# 얼리 스타핑, 모델체크포인트 적용/ val 적용

# acc: 0.9892500042915344
# LSTM acc: 0.10262499749660492
# conv1d acc: 0.8792499899864197
