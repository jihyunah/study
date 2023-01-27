from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()

                                                # 인풋은 (60000, 5, 5, 1)
model.add(Conv2D(filters=10, kernel_size=(2,2), # 2,2짜리 10장을 만들겠다는 것. 
                #  이 필터의 갯수의 기준은 잘나올 것 같은 것으로 사람이 정하는 것. 
                 input_shape=(10, 10, 1))) # --> 원래 이미지의 사이즈    # --> (N,4,4,10)  # N은 행 무시기에 통상적으로 NONE을 쓰는 것임. 훈련에 상관안줌.
      # (batch_size, rows, columns, channels)
      # batch_size는 훈련의 수, 단위이다. 그러니까 행과 똑같은 것이다. 
model.add(Conv2D(5, (2,2)))     # --> (N,3,3,5) #--> filters와 kerner_size는 생략해도 됨. 
model.add(Conv2D(7, (2,2)))     # --> (N,3,3,5) #--> filters와 kerner_size는 생략해도 됨. 
model.add(Conv2D(6, 2))     # 2로만 써도 커널 사이즈 2*2로 받아들인다. 
model.add(Flatten()) # --> 위에 애들이 쫙 펴진다.    # --> (N,45)
model.add(Dense(units=10))                                # --> (N,10)
         # 인풋은 (batch_size, input_dim)
model.add(Dense(4, activation='relu'))                               #--> (N,1)  # 지현,성환,건률,렐루

model.summary()




