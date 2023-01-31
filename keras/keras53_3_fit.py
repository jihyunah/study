import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator #발전기
from tensorflow.keras.callbacks import EarlyStopping



#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255, # --> minmaxscale 하겠다는 뜻. 
    horizontal_flip=True, # --> 수평 가로로 뒤집겠다는 뜻. 
    vertical_flip=True, # --> 수직 세로로 뒤집겠다는 뜻.
    width_shift_range=0.1, # --> 너비, 가로로 0.1 만큼 이동시키겠다
    height_shift_range=0.1, # --> 높이 길이 세로로 0.1만큼 이동시키겠다
    rotation_range=5, # --> 지정한 값만큼 회전시킨다. 최대값은 180도
    zoom_range=1.2, # --> 확대 
    shear_range=0.7, # --> 기울임
    fill_mode='nearest' # --> 빈자리를 채워줌
)


test_datagen = ImageDataGenerator(
    rescale=1./255      # 테스트 데이터는 rescale만 한다. 왜? \
        # 테스트 데이터의 목적은 평가하기 위한 데이터이기 때문에 정확한 평가를 위해 증폭하지 않은 원가 데이터를 쓴다. 
)

xy_train = train_datagen.flow_from_directory('./_data/brain/train/', target_size=(100, 100), # 이렇게 하면 test의 두 파일이 0과 1로 들어온다. # 이렇게 하면 모든 사진이 이 사이즈로 증폭된다. 
                                             batch_size=1000, # 배치 사이즈만큼씩 잘라서 훈련시킨다
                                             class_mode='binary', # 수치 
                                             color_mode='grayscale', 
                                             shuffle=True) # directory = folder # 폴더에 있는 이미지를 가져오겠다. train까지만 쓰면 ad, normal 다 들어온다. 
            # x = (160, 100, 100, 1)
            # y = (160, )
            #     0 = 80개 
            #     1 = 80개
            
            # Found 160 images belonging to 2 classes.
            
xy_test = test_datagen.flow_from_directory('./_data/brain/test/', target_size=(100, 100), # 이렇게 하면 test의 두 파일이 0과 1로 들어온다. # 이렇게 하면 모든 사진이 이 사이즈로 증폭된다. 
                                             batch_size=1000, # 배치 사이즈만큼씩 잘라서 훈련시킨다
                                             class_mode='binary', # 수치 
                                             color_mode='grayscale', 
                                             shuffle=True ) # directory = folder # 폴더에 있는 이미지를 가져오겠다. train까지만 쓰면 ad, normal 다 들어온다. 
#                                            0과 1이 적절히 섞여야 하기 때문에 셔플이 중요. 
            # x = (120, 150, 150, 1)
            # y = (120, )
            #     0 = 60개 
            #     1 = 60개
            
            # Found 120 images belonging to 2 classes.

#2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(128, (2,2), activation='relu', input_shape=(100, 100, 1)))
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

hist = model.fit(xy_train[0][0], xy_train[0][1], epochs=100, # 1 에포당 배치 사이즈만큼 걷는거 총 몇번 하느냐. 
                    validation_data=(xy_test[0][0], xy_test[0][1]),
                    batch_size=10, callbacks=[es])  # x, y, batchsize가 들어가있음 
#                     validation_split로도 나눌 수 있음. 그럼 테스트 데이터가 아니라 트레인 데이터에서 나누어지게 됨. 

# fit generator 대신에 fit도 쓸 수 있다. 근데 full 배치로 잡아놓고 사용해야됨. 

# hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=100, # step per epoch = train 데이터 수 나누기 배치 사이즈 
#                     validation_data=xy_test, validation_steps=4, )  # x, y, batchsize가 들어가있음

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




# Epoch 00065: early stopping  batch 10일때
# loss: 8.89783405000344e-06
# val_loss: 1.1512545347213745
# accuracy: 1.0
# val_acc: 0.7583333253860474