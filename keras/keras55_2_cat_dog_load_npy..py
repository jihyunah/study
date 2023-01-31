import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator #발전기
from sklearn.model_selection import train_test_split


x_train = np.load('c:/_data/cat_dog/train/catdog_x_train.npy')
y_train = np.load('c:/_data/cat_dog/train/catdog_y_train.npy')
x_test = np.load('c:/_data/cat_dog/test/catdog_x_test.npy')
y_test = np.load('c:/_data/cat_dog/test/catdog_y_test.npy')

# x_test = np.load('./_data/brain/brain_x_test.npy')
# y_test = np.load('./_data/brain/brain_y_test.npy')
train_datagen = ImageDataGenerator(
    rescale=1./255      # 테스트 데이터는 rescale만 한다. 왜? \
        # 테스트 데이터의 목적은 평가하기 위한 데이터이기 때문에 정확한 평가를 위해 증폭하지 않은 원가 데이터를 쓴다. 
)

test_datagen = ImageDataGenerator(
    rescale=1./255      # 테스트 데이터는 rescale만 한다. 왜? \
        # 테스트 데이터의 목적은 평가하기 위한 데이터이기 때문에 정확한 평가를 위해 증폭하지 않은 원가 데이터를 쓴다. 
)

xy_test = test_datagen.flow_from_directory('c:/_data/cat_dog/test/', target_size=(200, 200), # 이렇게 하면 test의 두 파일이 0과 1로 들어온다. # 이렇게 하면 모든 사진이 이 사이즈로 증폭된다. 
                                             batch_size=13000, # 배치 사이즈만큼씩 잘라서 훈련시킨다
                                             class_mode='binary', # 
                                             color_mode='rgb', 
                                             shuffle=True) # directory = folder # 폴더에 있는 이미지를 가져오겠다. train까지만 쓰면 ad, normal 다 들어온다. 
            # x = (12500, 200, 200, 1)
            # y = (12500, )
            #     0 = 개 
            #     1 = 개
            
            # Found 12500 images belonging to 1 classes.
            
dog = train_datagen.flow_from_directory('c:/_data/cat_dog/dog/', target_size=(200, 200), # 이렇게 하면 test의 두 파일이 0과 1로 들어온다. # 이렇게 하면 모든 사진이 이 사이즈로 증폭된다. 
                                             # 배치 사이즈만큼씩 잘라서 훈련시킨다
                                             class_mode=None, # 
                                             color_mode='rgb', 
                                             shuffle=True)
            



# print(x_train.shape) # (25000, 200, 200, 1)
# print(y_train.shape) # (25000,)
# print(xy_test[0][0].shape) # (12500, 200, 200, 3)
# print(x_train[100])

print(dog[0][0].shape) # (200,200,3)



#2. 모델구성
model = Sequential()
model.add(Conv2D(128, (2,2), padding='same', activation='relu', input_shape=(200, 200, 3)))
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
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
                    validation_data=[x_test, y_test], 
                    batch_size=16, callbacks=[es])  # x, y, batchsize가 들어가있음

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

loss2 = model.predict(dog)
print(loss2)
# result = model.predict(xy_test[0])


# Epoch 00051: early stopping
# loss: 0.6889081001281738
# val_loss: 0.716422975063324
# accuracy: 0.6000000238418579
# val_acc: 0.0
# [[0.5001117]]