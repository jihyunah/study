import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator #발전기
from tensorflow.keras.datasets import fashion_mnist

(x_train,y_train), (x_test, y_test) = fashion_mnist.load_data()
augument_size=100  # -->100장으로 증폭하겠다. 

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

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28 ,28, 1),  # x
    np.zeros(augument_size),                                                   # y
    batch_size=augument_size, 
    shuffle=True
    )
print(x_data[0])
print(x_data[0][0].shape) # (100, 28, 28, 1)
print(x_data[0][1].shape) # (100, )

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49): 
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray')
plt.show()


            
xy_test = test_datagen.flow_from_directory('./_data/brain/test/', target_size=(200, 200), # 이렇게 하면 test의 두 파일이 0과 1로 들어온다. # 이렇게 하면 모든 사진이 이 사이즈로 증폭된다. 
                                             batch_size=10, # 배치 사이즈만큼씩 잘라서 훈련시킨다
                                             class_mode='binary', # 수치 
                                             color_mode='grayscale', 
                                             shuffle=True) # directory = folder # 폴더에 있는 이미지를 가져오겠다. train까지만 쓰면 ad, normal 다 들어온다. 
            # x = (120, 150, 150, 1)
            # y = (120, )
            #     0 = 60개 
            #     1 = 60개
            
            # Found 120 images belonging to 2 classes.

