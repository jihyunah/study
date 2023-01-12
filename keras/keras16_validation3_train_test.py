import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

# 실습! 잘라봐. 
# train_test_split로 자르기
# 10:3:3 으로 나눠라

x_train, x_input, y_train, y_input = train_test_split(x, y, train_size=0.65, shuffle=False)
x_test, x_val, y_test, y_val = train_test_split(x_input, y_input, train_size=0.5, shuffle=False)

print(x_train, x_test, x_val)
print(y_train, y_test, y_val)





'''
x_train = np.array(range(1,11))
y_train = np.array(range(1,11))
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])
x_val = np.array([14,15,16]) # 교과서 문제 풀고, 답안지 보며 답 확인하는 느낌/ 훈련-->검증-->훈련-->검증
y_val = np.array([14,15,16])
'''
'''
#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_validation, y_validation))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

result = model.predict([17])
print('17의 예측값 :', result)
'''