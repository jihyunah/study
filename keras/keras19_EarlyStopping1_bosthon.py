from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (506, 13) (506,) --> 행 무시, 열 우선!!

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, test_size=0.2)

#2. 모델구성
model = Sequential()
# model.add(Dense(5, input_dim=13))
model.add(Dense(100, input_shape=(13,), activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')

from tensorflow.keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True,
                              verbose=1) # 발로스를 기준으로 최소값 갱신 5번까지 기다리고 멈추겠다. 그리고 최소로스의 최적 웨이트에서 로스값을 구하겠다. 
hist = model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_split=0.2, callbacks=[earlyStopping], verbose=1) 


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

print("===================================")
print(hist) # <keras.callbacks.History object at 0x000002593F1E46A0>
print("===================================")
print(hist.history['loss']) #loss 값을 리스트로 정리해놓음. ''으로 되어있는건 키, []로 리스트 형태로 묶인건 밸류. 이 전체의 형태를 딕셔너리 형태. 
#그래서 이 딕셔너리는 키와 밸류로 구성되어 있다. 리스트는 두개 이상의 형태. 
print("===================================")
print(hist.history['val_loss']) 

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6)) #그림판의 사이즈 정하는 것
plt.plot(hist.history['loss'], c='red', marker='.', label='loss') #plot은 선 긋는 것, x가 어차피 에포 순대로 들어가기 때문에 y만 적어주면 됨. 
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss') #c는 컬러, marker는 선의 형태, 라벨은 선의 이름
plt.grid() # 격자, 모눈종이 형태로 보일 것임
plt.xlabel('epochs') #x축의 이름 설정
plt.ylabel('loss')
plt.title('boston loss')
plt.legend() # 선의 이름을 보여줌
# plt.legend(loc='upper left') # loc는 location 위치의 줄임말
plt.show()
