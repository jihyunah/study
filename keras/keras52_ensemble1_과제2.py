# csv 파일의 날짜를 오름차순으로 변환하기 0
# 여러개의 컬럼중에 5개 임의로 고를 것 0 --> 시가, 종가, 거래량, 고가, 저가 
# 앙상블 소스와 가중치를 제출 --> 가중치는 모델체크포인트로 파일로 해서 내기.   --> 삼성 월요일 시가를 맞추기. 
# 메일 제목 : 이지현 78,700원
# 1등 ~ 5등까지 시상. 


import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Flatten, Input
from tensorflow.python.keras.layers import Conv1D, Dropout, MaxPooling1D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
 
#1. 데이터
path = '/Applications/study/keras/_data/samsung_jusik/'
samsung_csv = pd.read_csv(path + 'samsung.csv', index_col=0, encoding='cp949')
amore_csv = pd.read_csv(path + 'amore.csv', index_col=0, encoding='cp949')
# submission = pd.read_csv(path + 'submission.csv', index_col=0)

samsung_csv = samsung_csv.sort_index()  # index를 오름차순으로 바꿔준다. 
amore_csv = amore_csv.sort_index()
print(amore_csv)


print(samsung_csv)
print(samsung_csv.shape) # [1980 rows x 16 columns]
print(samsung_csv.columns) # ['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
    #    '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
    #   dtype='object')
    
samsung_csv = samsung_csv[['거래량', '고가', '저가', '종가', '시가', 
                           '전일비', 'Unnamed: 6', '등락률', '금액(백만)', '신용비', 
                           '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']]
amore_csv = amore_csv[['거래량', '고가', '저가', '종가', '시가', '전일비', 'Unnamed: 6', 
                       '등락률', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', 
                       '프로그램', '외인비']]


print(samsung_csv.info()) # 결측치 있는 것을 확인할 수 있음. 

samsung_csv = samsung_csv.dropna() # samsung_csv의 결측치 모두 삭제
samsung_csv['시가'] = pd.to_numeric(samsung_csv['시가'].str.replace(',',''))
samsung_csv['시가'] = samsung_csv['시가'].astype(np.float64)
samsung_csv['종가'] = pd.to_numeric(samsung_csv['종가'].str.replace(',',''))
samsung_csv['종가'] = samsung_csv['종가'].astype(np.float64)
samsung_csv['고가'] = pd.to_numeric(samsung_csv['고가'].str.replace(',',''))
samsung_csv['고가'] = samsung_csv['고가'].astype(np.float64)
samsung_csv['저가'] = pd.to_numeric(samsung_csv['저가'].str.replace(',',''))
samsung_csv['저가'] = samsung_csv['저가'].astype(np.float64)
samsung_csv['거래량'] = pd.to_numeric(samsung_csv['거래량'].str.replace(',',''))
samsung_csv['거래량'] = samsung_csv['거래량'].astype(np.float64)

print(amore_csv)
print(amore_csv.shape) # [2220 rows x 16 columns]
print(amore_csv.columns)  # ['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', 
# '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']
print(amore_csv.info())
amore_csv = amore_csv.dropna()  # amore_csv의 결측치 모두 삭제 

print(samsung_csv.info())
print(amore_csv.info())    # 결측치 삭제된 것을 확인할 수 있음

# 1.3) data 개수 통일 (train,test split을 위한)
amore_csv = samsung_csv.sample(n=len(samsung_csv), random_state=42)
print(amore_csv.shape) #(1977, 16)
print(samsung_csv.shape) #(1977, 16)

for i,col in enumerate(samsung_csv.columns): # 컬럼 인덱스 확인
    print(i,col)
for i,col in enumerate(amore_csv.columns): # 컬럼 인덱스 확인
    print(i,col)
    
samsung_csv.drop(samsung_csv.columns[[5,6,7,8,9,10,11,12,13,14,15]], 
                 axis=1,inplace=True)
print(samsung_csv)
print(samsung_csv.info())

amore_csv.drop(amore_csv.columns[[5,6,7,8,9,10,11,12,13,14,15]], 
               axis=1,inplace=True)


samsung_x = samsung_csv[['거래량', '고가', '저가', '종가', '시가']]
samsung_y = samsung_csv[['시가']].to_numpy()

print(samsung_x.shape) # (1977, 5)
print(samsung_y.shape) # (1977, 1)

samsung_csv = MinMaxScaler().fit_transform(samsung_csv)
amore_csv = MinMaxScaler().fit_transform(amore_csv)


def split_data(dataset, timesteps):
    tmp = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        tmp.append(subset)
    return np.array(tmp)

samsung_x = split_data(samsung_csv, 5)
amore_x = split_data(amore_csv, 5) # 
print(samsung_x.shape) #(1973, 5, 5)
print(amore_x.shape) #(1973, 5, 5)

samsung_y = samsung_y[4:, :] # x 데이터와 shape을 맞춰주기 위해 4개 행 제거
print(samsung_y.shape) #(1973, 1)

# 예측 데이터 추출 (마지막 값)
samsung_x_predict = samsung_x[-1].reshape(-1, 5, 5)
amore_x_predict = amore_x[-1].reshape(-1, 5, 5)
samsung_x_predict = samsung_x_predict
amore_x_predict = amore_x_predict
print(samsung_x_predict.shape) # (1, 5, 5)
print(amore_x_predict.shape) # (1, 5, 5)
# print(samsung_x_predict)
# print(amore_x_predict)


samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test, amore_x_train, amore_x_test = train_test_split(
    samsung_x, samsung_y, amore_x, train_size=0.8, random_state=1234)

print(samsung_x_train.shape, samsung_x_test.shape)  # (1578, 5, 5) (395, 5, 5)
print(samsung_y_train.shape, samsung_y_test.shape) # (1578, 1) (395, 1)
print(amore_x_train.shape, amore_x_test.shape)  # (1578, 5, 5) (395, 5, 5)



# x1_train = x1_train.reshape(1581, 5, 1)
# x1_test = x1_test.reshape(396, 5, 1)
# x2_train = x2_train.reshape(1581, 5, 1)
# x2_test = x2_test.reshape(396, 5, 1)

# print(x1_train.shape) 
# print(x1_test.shape)
# print(x2_train.shape)
# print(x2_test.shape)



#2. 모델구성

#2-1. 모델 1 삼성.
input1 = Input(shape=(5, 5))
dense1 = Conv1D(128, 1, padding='same', activation='relu', name='ds11')(input1)
dense2 = Conv1D(64, 1, padding='same', activation='relu', name='ds12')(dense1)
dense3 = Conv1D(32, 1, padding='same', activation='relu', name='ds13')(dense2)
dense5 = Conv1D(16, 1, padding='same', activation='relu', name='ds14')(dense3)
dense6 = Flatten(name='ds28')(dense5)
output1 = Dense(32, activation='relu', name='ds29')(dense6)   # concatenate 의 input

# 머지 할 것이기 때문에 아직 output 내지 않음. 

#2-2. 모델 2.
input2= Input(shape=(5, 5))
dense21= Conv1D(128, 1, padding='same', activation='relu', name='ds31')(input2)
dense22= Conv1D(64, 1, padding='same', activation='relu', name='ds32')(dense21)
dense23= Conv1D(32, 1, padding='same', activation='relu', name='ds34')(dense22)
dense33= Conv1D(16, 1, padding='same', activation='relu', name='ds47')(dense23)
dense34= Flatten(name='ds49')(dense33)
output2= Dense(32, activation='relu', name='ds50')(dense34)
# dense21= Dense(21, activation='linear', name='ds21')(input2)
# dense22= Dense(22, activation='linear', name='ds22')(dense21)
# output2= Dense(23, activation='linear', name='ds23')(dense22)  # concatenate 의 input

#2-3. 모델병합
from tensorflow.python.keras.layers import concatenate  # concatenate 사슬같이 잇다; 연쇄시키다;
merge1 = concatenate([output1, output2], name='mg1') # 2개 이상은 리스트 형태
merge2 = Dense(32, activation='relu', name='mg2')(merge1)
merge3 = Dense(16, activation='relu', name='mg3')(merge2)
merge5 = Dense(8, activation='relu', name='mg4')(merge3)
last_output = Dense(1, name='last')(merge5)

model = Model(inputs=[input1, input2], outputs=last_output)  # 모델이 두개라서 인풋이 2개임. 

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', 
                   restore_best_weights=True, patience=100, verbose=1)

import datetime
date=datetime.datetime.now()

date=date.strftime("%m%d_%H%M")

mcp = ModelCheckpoint(
    filepath='/Applications/study/keras/_save/MCP/' + 'k52_ensemble2_hw2' + date + '_{epoch}-{val_loss:.4f}.h5',
    monitor='val_loss',
    verbose=2,
    save_best_only=True
)



model.fit([samsung_x_train, amore_x_train], samsung_y_train, 
          epochs=400, batch_size=1, validation_split=0.3, callbacks=[es, mcp])

# model.save_weights('/Applications/study/keras/_save/' + 'stock_weight.h5') # 가중치만 저장

#4. 평가, 예측
loss = model.evaluate([samsung_x_test, amore_x_test], samsung_y_test)
print('loss:', loss)

result = model.predict([samsung_x_predict, amore_x_predict])
print('1/30 삼성전자 시가 예측값:', result)




# 날짜를 순서대로 변환. 
# 여러개의 컬럼중에 5개 임의로 고를 것. 
# 앙상블 소스와 가중치를 제출 --> 가중치는 모델체크포인트로 파일로 해서 내기.   --> 삼성 월요일 시가를 맞추기. 
# 메일 제목 : 이지현 78,700원
# 1등 ~ 5등까지 시상. 

# 64600원
# 드롭아웃 없고, randomstate 1234 scaler 안했을때 loss: [67298720.0, 0.0] 1/30 삼성전자 시가 예측값: [[63401.195]] --> 파일이름 k52_ensemble2_hw20129_1936_400-54481472.0000.h5
# 드롭아웃 없고, randomstate 333 scaler 했을 때 loss: [75523600.0, 0.0] 1/30 삼성전자 시가 예측값: [[65252.805]] --> 파일이름 k52_ensemble2_hw20129_2002_397-1694586496.0000.h5
# 드롭아웃 없고, randomstate 333 scaler 안했을 때 loss: [61026808.0, 0.0] 1/30 삼성전자 시가 예측값: [[65167.32]] --> 파일이름 k52_ensemble2_hw20129_2019_400-1072128704.0000.h5
# 드롭아웃 없고, randomstate 1234 scaler 했을 때 loss: loss: [41608084.0, 0.0] 1/30 삼성전자 시가 예측값: [[64523.332]] --> 파일이름 k52_ensemble2_hw20129_2035_382-29967208.0000.h5