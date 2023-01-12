import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
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

#2. 모델구성 (순차형)
# model = Sequential()
# model.add(Dense(130, activation='relu', input_shape=(64,)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.summary() # Total params: 36,308


#2. 모델구성 (함수형)
input1 = Input(shape=(64,))
dense1 = Dense(130, activation='relu')(input1)
dense2 = Dense(128, activation='relu')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
dense4 = Dense(32, activation='relu')(dense3)
dense5 = Dense(16, activation='relu')(dense4)
dense6 = Dense(8, activation='relu')(dense5)
output1 = Dense(10, activation='softmax')(dense6)
model = Model(inputs=input1, outputs=output1)
model.summary() # Total params: 36,308



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor='accuracy', mode='max', patience=30, 
                              restore_best_weights=True, 
                              verbose=1)
hist = model.fit(x_train, y_train, epochs=10, batch_size=1, 
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