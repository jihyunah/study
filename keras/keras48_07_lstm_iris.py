from sklearn.datasets import load_iris # 어떤 꽃인지 분류하는 것
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
datasets = load_iris()
print(datasets.DESCR) # 정보 설명 //  판다스 .describe() / .info()
print(datasets.feature_names) # 판다스 .columns

x = datasets.data
y = datasets['target']

# y = pd.get_dummies(y) # 원핫인코딩 한 것임. 총 3가지 방법


#다른 방법으로는 1):
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

# 2) from sklearn.preprocessing import OneHotEncoder
# import numpy as np

# onehot = OneHotEncoder()
# onehot.fit(y.reshape(-1, 1))
# y = onehot.transform(y.reshape(-1, 1)).toarray()

#print(x)
print(y)
print(x.shape, y.shape) # (150, 4), (150,3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, # False의 문제점은 섞지 않게 되니까 하나의 클래스의 동일한 놈들이 TEST로 다 짤려나가서 성능이 적어짐.
     random_state=333, 
     test_size=0.2,
     stratify=y)   # 통계적인이란 뜻. 그래서 분류에서만 쓸 수 있음
# print(y_train)
# print(y_test)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x_train) # --> 가중치를 생성시킨다는 것
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test) 

print(x_train.shape, x_test.shape) # (120, 4) (30, 4)
x_train = x_train.reshape(120, 2, 2, 1)
x_test = x_test.reshape(30, 2, 2, 1)



#2. 모델구성 (순차형)
model = Sequential()
model.add(Conv2D(128, (2,2), padding='same', activation='relu', input_shape=(2, 2, 1)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax')) # 다중분류의 마지막 액티베이션은 softmax이고, 아웃풋 노드는 y의 클래스 개수와 같게 3.
model.summary() # Total params: 11,820





#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True, 
                              verbose=1)

# import datetime 
# date = datetime.datetime.now()
# date = date.strftime('%M%D_%H%M')

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1, 
#                       filepath= filepath + 'k31_07_' + date + '_' + filename)


model.fit(x_train, y_train, epochs=300, batch_size=1, 
          validation_split=0.2, 
          verbose=1, callbacks=[earlyStopping])


#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', accuracy)

# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
# print('y_predict:', y_predict)

# print(y_predict.shape)
print(y_test)
print(y_test.shape)

y_test = np.argmax(y_test, axis=1)
print('y_test:', y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy score :', acc)

# accuracy score : 0.9666666666666667

'''
 
'''

# CNN  accuracy score : 0.9333333333333333
