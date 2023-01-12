from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()

model.add(Conv2D(filters=10, kernel_size=(2,2), # 2,2짜리 10장을 만들겠다는 것. 
                #  이 필터의 갯수의 기준은 잘나올 것 같은 것으로 사람이 정하는 것. 
                 input_shape=(5, 5, 1))) # --> 원래 이미지의 사이즈 
model.add(Conv2D(filters=5, kernel_size=(2,2)))
model.add(Flatten()) # --> 위에 애들이 쫙 펴진다.
model.add(Dense(10))
model.add(Dense(1))

model.summary()




