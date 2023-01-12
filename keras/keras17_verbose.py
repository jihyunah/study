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
model.add(Dense(5, input_shape=(13,))) # --> 다차원이 나올 때 사용. ex) (100,10,5)이면 잇풋 쉐이프에 (10,5)들어가야함. 
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer = 'adam')
start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=1,
          validation_split=0.2,
          verbose=3) #verbose? 1은 true.
#0은 입 닥쳐. 훈련하는 결과를 보는건 편하다. 그러나 보여주는 것 자체로 딜레이가 생긴다. 
# verbose = 2하면 프로그레스바 (진행바)가 안보인다. 
# 3일 때는 에포 숫자만 볼 수 있음. 3이상부터는 3과 동일하다. 
end = time.time()


#3. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

print("걸린시간:", end - start)

# verbos = 1일 때 걸린시간: 11.764618396759033
# verbos = 0일 때 걸린시간: 9.528396129608154
