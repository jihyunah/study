from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#1. 데이터
datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DESCR)
#print(datasets.feature_names) 


x = datasets['data']
y = datasets['target']
print(x.shape, y.shape) # (569, 30) (569,) 

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, test_size=0.2)

#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='linear', input_shape=(30,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # y값이 0과 1이기 때문에.

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy']) 
# 이진분류의 로스는 'binary_crossentropy' 뿐이다.

from tensorflow.keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True,
                              verbose=1)

model.fit(x_train, y_train, epochs=10, batch_size=16,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)

#4. 평가, 예측
'''loss = model.evaluate(x_test, y_test)
print('loss, accuracy :', loss) # 로스랑 애큐러시가 나옴
'''

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
print("accuracy_score :", acc) # accuracy_score : 0.9473684210526315


