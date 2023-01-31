import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator #발전기


# np.save('./_data/brain/brain_x_train.npy', arr=xy_train[0][0])
# np.save('./_data/brain/brain_y_train.npy', arr=xy_train[0][1])
# # np.save('./_data/brain/brain_xy_train.npy', arr=xy_train[0])
# np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])
# np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0][1])  --> 경로 확인용으로 남겨둔 것임. 

x_train = np.load('./_data/brain/brain_x_train.npy')
y_train = np.load('./_data/brain/brain_y_train.npy')
x_test = np.load('./_data/brain/brain_x_test.npy')
y_test = np.load('./_data/brain/brain_y_test.npy')


print(x_train.shape, x_test.shape) # (160, 200, 200, 1) (120, 200, 200, 1)
print(y_train.shape, y_test.shape) # (160,) (120, 2)
# print(x_train[100])





#2. 모델구성
model = Sequential()
model.add(Conv2D(128, (2,2), activation='relu', input_shape=(200, 200, 1)))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'])

# 이진분류의 로스는 'binary_crossentropy' 뿐이다.

es = EarlyStopping(monitor='val_acc', mode='max', 
                   restore_best_weights=True, patience=50, verbose=1)

hist = model.fit(x_train, y_train, epochs=100, # 1 에포당 배치 사이즈만큼 걷는거 총 몇번 하느냐. 
                    validation_split=0.2, 
                    batch_size=10, callbacks=[es])  # x, y, batchsize가 들어가있음

# hist = model.fit_generator(x_train, y_train, steps_per_epoch=1, epochs=100, # step per epoch = train 데이터 수 나누기 배치 사이즈
#                     validation_data= (x_test, y_test), validation_steps=12, callbacks=[es] ) 
#                         --> fit 제너레이터는 xy 합쳐야 돌아간다. 

# fit generator 대신에 fit도 쓸 수 있다. 근데 full 배치로 잡아놓고 사용해야됨. 

accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss:', loss[-1])  
print('val_loss:', val_loss[-1]) # 가장 끝 값 나옴
print('accuracy:', accuracy[-1])
print('val_acc:', val_acc[-1])

# matplotlib 으로 그림 그려라 

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6)) #그림판의 사이즈 정하는 것
plt.plot(hist.history['loss'], c='red', marker='.', label='loss') #plot은 선 긋는 것, x가 어차피 에포 순대로 들어가기 때문에 y만 적어주면 됨. 
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss') #c는 컬러, marker는 선의 형태, 라벨은 선의 이름
plt.plot(hist.history['acc'], c='green', marker='.', label='acc') #c는 컬러, marker는 선의 형태, 라벨은 선의 이름
plt.plot(hist.history['val_acc'], c='purple', marker='.', label='val_acc') #c는 컬러, marker는 선의 형태, 라벨은 선의 이름
plt.grid() # 격자, 모눈종이 형태로 보일 것임
plt.xlabel('epochs') #x축의 이름 설정
plt.ylabel('loss, accuracy')
plt.title('brain_loss&accuracy')
plt.legend() # 선의 이름을 보여줌
# plt.legend(loc='upper left') # loc는 location 위치의 줄임말
plt.show()