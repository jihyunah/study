import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv1D, LSTM
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(178, 13) (178,)
# print(y)
# print(np.unique(y)) # [0 1 2] --> y는 0,1,2 만 있다는 것
# print(np.unique(y, return_counts=True)) 
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))


# -->one hot encoding의 3가지 방법 
# 1) from sklearn.preprocessing import OneHotEncoder
# import numpy as np

# onehot = OneHotEncoder()
# onehot.fit(y.reshape(-1, 1))
# y = onehot.transform(y.reshape(-1, 1)).toarray()

# # 2) y = pd.get_dummies(y) --> 오류 뜸


from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

print(y.shape) #(178,3)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, 
                                                    train_size=0.8, stratify=y)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train) # --> 가중치를 생성시킨다는 것
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

print(x_train.shape, x_test.shape) # (142, 13) (36, 13)
x_train = x_train.reshape(142, 13, 1)
x_test = x_test.reshape(36, 13, 1)


#2. 모델구성 (순차형)
model = Sequential()
model.add(Conv1D(128, 2, padding='same', activation='relu', input_shape=(13, 1)))
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(Conv1D(16, 2, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='linear'))
model.add(Dense(3, activation='softmax')) # 다중분류의 마지막 액티베이션은 softmax이고, 아웃풋 노드는 y의 클래스 개수와 같게 3.
model.summary() # Total params: 29,615




#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, 
                              verbose=1)

# import datetime 
# date = datetime.datetime.now()
# date = date.strftime('%M%D_%H%M')

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1, 
#                       filepath= filepath + 'k31_08_' + date + '_' + filename)


model.fit(x_train, y_train, epochs=300, batch_size=1,
           validation_split=0.2, verbose=1, callbacks=[earlyStopping])

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



'''


'''
# cnn accuracy score : 0.8888888888888888
# LSTM accuracy score : 0.9722222222222222
# conv1d accuracy score : 0.9166666666666666