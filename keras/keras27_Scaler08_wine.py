import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping


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

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x_train) # --> 가중치를 생성시킨다는 것
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test) 

#2. 모델구성
model = Sequential()
model.add(Dense(130, activation='relu', input_shape=(13,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='linear'))
model.add(Dense(3, activation='softmax')) # 다중분류의 마지막 액티베이션은 softmax이고, 아웃풋 노드는 y의 클래스 개수와 같게 3.

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, 
                              verbose=1)
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
loss: 0.09246271848678589
accucary: 0.9722222089767456
y_predict: [1 0 1 0 1 1 1 0 0 1 1 1 2 0 2 1 2 0 1 0 1 2 0 0 0 0 0 2 2 2 1 2 2 0 2 1]
y_test: [1 0 1 0 1 1 1 0 0 1 1 1 2 0 2 1 2 1 1 0 1 2 0 0 0 0 0 2 2 2 1 2 2 0 2 1]
accuracy score : 0.9722222222222222


minmax scaler:
accuracy score : 0.9444444444444444


standard scaler:
accuracy score : 0.9444444444444444

'''