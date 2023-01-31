# 가위 바위 보 모델 만들엇. 

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator #발전기
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

train_datagen = ImageDataGenerator(
    rescale=1./255, # --> minmaxscale 하겠다는 뜻. 
    # horizontal_flip=True, # --> 수평 가로로 뒤집겠다는 뜻. 
    # vertical_flip=True, # --> 수직 세로로 뒤집겠다는 뜻.
    # width_shift_range=0.1, # --> 너비, 가로로 0.1 만큼 이동시키겠다
    # height_shift_range=0.1, # --> 높이 길이 세로로 0.1만큼 이동시키겠다
    # rotation_range=5, # --> 지정한 값만큼 회전시킨다. 최대값은 180도
    # zoom_range=1.2, # --> 확대 
    # shear_range=0.7, # --> 기울임
    # fill_mode='nearest' # --> 빈자리를 채워줌
)     # 저장할 때 원본 가지고 있는게 낫기에 스케일링만 해서 저장해줌. 


# test_datagen = ImageDataGenerator(
#     rescale=1./255      # 테스트 데이터는 rescale만 한다. 왜? \
        # 테스트 데이터의 목적은 평가하기 위한 데이터이기 때문에 정확한 평가를 위해 증폭하지 않은 원가 데이터를 쓴다. 
# )

xy_train_test = train_datagen.flow_from_directory('c:/_data/rps/rps/', target_size=(200, 200), # 이렇게 하면 test의 두 파일이 0과 1로 들어온다. # 이렇게 하면 모든 사진이 이 사이즈로 증폭된다. 
                                             batch_size=3000, # 배치 사이즈만큼씩 잘라서 훈련시킨다
                                             class_mode='categorical', # 원핫 안해도 됨. 원본 수치로 저장할거니까. 
                                             color_mode='rgb', 
                                             shuffle=True) # directory = folder # 폴더에 있는 이미지를 가져오겠다. train까지만 쓰면 ad, normal 다 들어온다. 
            # x = (25000, 200, 200, 1)
            # y = (25000, )
            #     0 = 개 
            #     1 = 개
            
            # Found 2520 images belonging to 3 classes.
            
# xy_test = test_datagen.flow_from_directory('c:/_data/cat_dog/test/test1', target_size=(200, 200), # 이렇게 하면 test의 두 파일이 0과 1로 들어온다. # 이렇게 하면 모든 사진이 이 사이즈로 증폭된다. 
#                                              batch_size=1000, # 배치 사이즈만큼씩 잘라서 훈련시킨다
#                                              class_mode='categorical', # 
#                                              color_mode='rgb', 
#                                              shuffle=True) # directory = folder # 폴더에 있는 이미지를 가져오겠다. train까지만 쓰면 ad, normal 다 들어온다. 
            # x = (12500, 200, 200, 1)
            # y = (12500, )
            #     0 = 개 
            #     1 = 개
            
            # Found 12500 images belonging to 1 classes.



# print(xy_train_test)
# <keras.preprocessing.image.DirectoryIterator object at 0x00000222A0DFA280>


# print(xy_train[0]) #xy_train의 0번째는 뭐야?
print(xy_train_test[0][0]) #xy_train의 0의 0번째
# print(xy_train[0][1])
print(xy_train_test[0][0].shape) #(2520, 200, 200, 3)
# print(xy_train_test[0][1]) #xy_train의 0의 1번째 # [1. 0. 0. 1. 0.] --> y값임. batch_size 수에 맞게 나옴. 
print(xy_train_test[0][1].shape) #(2520, 3)   -> binary 일 때 
# # # 배치 사이즈를 크게 잡으면 전체 데이터를 가져올 수 있음 --> (160, 200, 200, 1)
# print(type(xy_train[0]))  # 튜플 

x = xy_train_test[0][0] 
y = xy_train_test[0][1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, shuffle=True)

print(x_train.shape, y_train.shape) #(2016, 200, 200, 3) (2016, 3)
print(x_test.shape, y_test.shape) # (504, 200, 200, 3) (504, 3)



#2. 모델 구성 
model = Sequential()
model.add(Conv2D(128, (2,2), padding='same', activation='relu', input_shape=(200, 200, 3)))
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))


#3. 컴파일 훈련 

model.compile(loss='categorical_crossentropy', optimizer='adam') # 위에서 원핫 했기 때문에 

es = EarlyStopping(monitor='val_acc', mode='max', 
                   restore_best_weights=True, patience=50, verbose=1)

hist = model.fit(x_train, y_train, epochs=100, # 1 에포당 배치 사이즈만큼 걷는거 총 몇번 하느냐. 
                    validation_split=0.2, 
                    batch_size=32, callbacks=[es])  # x, y, batchsize가 들어가있음

# accuracy = hist.history['acc']
# val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss:', loss[-1])  
print('val_loss:', val_loss[-1]) # 가장 끝 값 나옴
# print('accuracy:', accuracy[-1])
# print('val_acc:', val_acc[-1])


# 평가 예측 
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)
print(y_predict)
print(y_predict.shape) #(6493, 1)

# loss: 1.0989956855773926 - 61에포 
# 51/51 [==============================] - 7s 142ms/step - loss: 1.0986 - val_loss: 1.0993
# loss: 1.0985718965530396
# val_loss: 1.0992618799209595