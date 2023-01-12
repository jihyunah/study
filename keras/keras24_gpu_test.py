import tensorflow as tf
print(tf.__version__)  # 2.7.4

gpus = tf.config.experimental.list_physical_devices('GPU')
# 구성된 경험적. 물리적 장치의 목록

print(gpus)

if(gpus):  # 만약 쥐피유가 돈다면 
    print("쥐피유 돈다")
else:    # 만약 쥐피유가 안돈다면
    print("쥐피유 안돈다")