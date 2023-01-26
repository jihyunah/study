from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#1. 데이터
datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DESCR)
#print(datasets.feature_names) 


x = datasets['data']
y = datasets['target']
print(x.shape, y.shape) # (569, 30) (569,) 

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, test_size=0.2)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train) # --> 가중치를 생성시킨다는 것
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

print(x_train.shape, x_test.shape) # (455, 30) (114, 30)
x_train = x_train.reshape(455, 5, 6, 1)
x_test = x_test.reshape(114, 5, 6, 1)

#2. 모델구성 (순차형)
model = Sequential()
model.add(Conv2D(128, (2,2), padding='same', activation='linear', input_shape=(5, 6, 1)))
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
model.add(Conv2D(16, (2,2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # y값이 0과 1이기 때문에.
model.summary() # Total params: 5,141





#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy']) 
# 이진분류의 로스는 'binary_crossentropy' 뿐이다.

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True,
                              verbose=1)

import datetime 
date = datetime.datetime.now()
date = date.strftime('%M%D_%H%M')

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1, 
                      filepath= filepath + 'k31_06_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=300, batch_size=16,
          validation_split=0.2,
          callbacks=[earlyStopping, mcp],
          verbose=1)

#4. 평가, 예측

# loss = model.evaluate(x_test, y_test)
# print('loss, accuracy :', loss) # 로스랑 애큐러시가 나옴


loss, accuracy = model.evaluate(x_test, y_test)
print('loss :', loss) # loss : 0.1719813495874405
print('accuracy :', accuracy) #  accuracy : 0.9473684430122375

y_predict = model.predict(x_test)
print(y_predict) 

# float  :  실수,  int :  정수

# print(y_test[:10])
# y_predict = np.round(y_predict)


y_predict = np.where(y_predict > 0.5, 1, 0)  # 정수로 반올림


from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print("accuracy_score :", acc)

# accuracy_score : 0.9473684210526315


'''
minmax scaler
accuracy_score : 0.9649122807017544


standard scaler
accuracy_score : 0.9649122807017544

drop out
accuracy_score : 0.9824561403508771

'''

# CNN accuracy_score : 0.9649122807017544
