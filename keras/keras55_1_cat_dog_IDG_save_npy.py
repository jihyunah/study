import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator #발전기

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


test_datagen = ImageDataGenerator(
    rescale=1./255      # 테스트 데이터는 rescale만 한다. 왜? \
        # 테스트 데이터의 목적은 평가하기 위한 데이터이기 때문에 정확한 평가를 위해 증폭하지 않은 원가 데이터를 쓴다. 
)

xy_train = train_datagen.flow_from_directory('c:/_data/cat_dog/train/', target_size=(200, 200), # 이렇게 하면 test의 두 파일이 0과 1로 들어온다. # 이렇게 하면 모든 사진이 이 사이즈로 증폭된다. 
                                             batch_size=20, # 배치 사이즈만큼씩 잘라서 훈련시킨다
                                             class_mode='binary', # 원핫 안해도 됨. 원본 수치로 저장할거니까. 
                                             color_mode='rgb', 
                                             shuffle=True) # directory = folder # 폴더에 있는 이미지를 가져오겠다. train까지만 쓰면 ad, normal 다 들어온다. 
            # x = (25000, 200, 200, 1)
            # y = (25000, )
            #     0 = 개 
            #     1 = 개
            
            # Found 25000 images belonging to 2 classes.
            
xy_test = test_datagen.flow_from_directory('c:/_data/cat_dog/test/', target_size=(200, 200), # 이렇게 하면 test의 두 파일이 0과 1로 들어온다. # 이렇게 하면 모든 사진이 이 사이즈로 증폭된다. 
                                             batch_size=20, # 배치 사이즈만큼씩 잘라서 훈련시킨다
                                             class_mode='binary', # 
                                             color_mode='rgb', 
                                             shuffle=True) # directory = folder # 폴더에 있는 이미지를 가져오겠다. train까지만 쓰면 ad, normal 다 들어온다. 
            # x = (12500, 200, 200, 1)
            # y = (12500, )
            #     0 = 개 
            #     1 = 개
            
            # Found 12500 images belonging to 1 classes.



print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x00000222A0DFA280>


# print(xy_train[0]) #xy_train의 0번째는 뭐야?
print(xy_train[0][0]) #xy_train의 0의 0번째
# print(xy_train[0][1])
print(xy_train[0][0].shape) #(25000, 200, 200, 1)
print(xy_train[0][1]) #xy_train의 0의 1번째 # [1. 0. 0. 1. 0.] --> y값임. batch_size 수에 맞게 나옴. 
print(xy_train[0][1].shape) #(25000,)   -> binary 일 때 
# # 배치 사이즈를 크게 잡으면 전체 데이터를 가져올 수 있음 --> (160, 200, 200, 1)
print(type(xy_train[0]))  # 튜플 


np.save('c:/_data/cat_dog//train/catdog_x_train.npy', arr=xy_train[0][0])
np.save('c:/_data/cat_dog/train/catdog_y_train.npy', arr=xy_train[0][1])
# np.save('./_data/brain/brain_xy_train.npy', arr=xy_train[0]) --> 이건 안된다. 넘파이 형태가 아닌 튜플형태다. 사용하려면 위의 두개를 append로 붙여야한다. 

np.save('c:/_data/cat_dog/test/catdog_x_test.npy', arr=xy_test[0][0])
np.save('c:/_data/cat_dog/test/catdog_y_test.npy', arr=xy_test[0][1])




# 원래 큰 이미지 파일들이 넘파이 형태로 저장됨. 


# print(type(xy_train)) # 데이터의 형태 : <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) # <class 'tuple'> = 리스트와 똑같다. 그러나 튜플은 한번 생성되면 바꿀 수 없다. 그럼 다른 변수로 빼서 수정하면 됨. 
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>



