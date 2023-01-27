from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, BatchNormalization, MaxPooling2D, Dropout, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)      # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)        # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))


x_train = x_train.reshape(50000, 32, 96)
x_test = x_test.reshape(10000, 32, 96)

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.8, random_state=333)

#2. 모델구성
model = Sequential()
model.add(Conv1D(128, 2, padding='same', input_shape=(32, 96), 
                 activation='relu'))  
model.add(Conv1D(64, 2, padding='same', activation='relu'))    
model.add(Conv1D(32, 2, padding='same', activation='relu'))           
model.add(Flatten())       
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3), 
#                  padding='same', strides=1))
# model.add(BatchNormalization())
# model.add(Conv2D(32, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', padding='same', strides=2))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(64, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

# input1 = Input(shape=(32, 32, 3))
# dense1 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(input1)
# dense2 = BatchNormalization()(dense1)
# dense3 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', strides=2)(dense2)
# dense4 = BatchNormalization()(dense3)
# dense5 = MaxPooling2D()(dense4)
# dense6 = Dropout(0.25)(dense5)
# dense7 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(dense6)
# dense8 = BatchNormalization()(dense7)
# dense9 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(dense8)
# dense10 = BatchNormalization()(dense9)
# dense11 = MaxPooling2D()(dense10)
# dense12 = Dropout(0.25)(dense11)
# dense13 = Flatten()(dense12)
# dense14 = Dense(512, activation='relu')(dense13)
# dense15 = Dropout(0.5)(dense14)
# output1 = Dense(10, activation='softmax')(dense15)
# model = Model(inputs=input1, outputs=output1)



#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=10, verbose=1)

# import datetime 
# date = datetime.datetime.now()
# date = date.strftime("%M%D_%H%M")

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1, 
#                       filepath= filepath + 'k34_2_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32, validation_data=[x_val, y_val], 
          callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss:', results[0])
print('acc:', results[1])

# acc: 0.8148750066757202
# LSTM acc: 0.09724999964237213
# conv1d acc: 0.09925000369548798