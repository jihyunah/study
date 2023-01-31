import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator #발전기
from tensorflow.keras.datasets import fashion_mnist

(x_train,y_train), (x_test, y_test) = fashion_mnist.load_data()
augument_size=40000  # -->100장으로 증폭하겠다. 
randidx = np.random.randint(x_train.shape[0], size=augument_size)
print(randidx)
print(len(randidx))   # 40000

x_augument = x_train[randidx].copy()
y_augument = y_train[randidx].copy()

print(x_augument.shape, y_augument.shape) # (40000, 28, 28) (40000,)
x_augument = x_augument.reshape(40000, 28, 28, 1)



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

x_augumented = train_datagen.flow(
   x_augument, 
   y_augument, # y
    batch_size=augument_size, 
    shuffle=True
    )
print(x_augumented[0][0].shape)  # (40000, 28, 28, 1)
print(x_augumented[0][1].shape)  # (40000,)

x_train = x_train.reshape(60000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented[0][0]))
y_train = np.concatenate((y_train, x_augumented[0][1])) 

print(x_train.shape, y_train.shape)  # (100000, 28, 28, 1) (100000,)