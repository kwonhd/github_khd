# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 00:45:57 2023

@author: gureh
"""
#%%
import tensorflow as tf
#%%
# TensorFlow로 간단한 linear regression을 구현 (new)
#H(x) = Wx+b
x_train=[1,2,3]
y_train=[1,2,3]

w = tf.Variable(tf.random.normal([1]),name='weight')
b = tf.Variable(tf.random.normal([1]),name='bias')

hyp = x_train*w+b
cost = tf.reduce_mean(tf.square(hyp - y_train))

print(hyp)
print(cost)
#%%
#미니마이즈
#t=[1.,2.,3.,4.]
#tf.reduce_mean(t) ==>2.5


#%%
import tensorflow as tf
import numpy as np

# 학습 데이터 생성
X = np.array([1, 2, 3], dtype=np.float32)
Y = np.array([1, 2, 3], dtype=np.float32)

# 가중치와 편향 초기화
W = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# 선형 회귀 모델 정의
def linear_regression(x):
    return W * x + b

# 손실 함수 정의
def mean_square_error(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# 경사 하강 옵티마이저 설정
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# 학습 루프 = train
for step in range(2001):
    # 자동 미분을 위해 tf.GradientTape 사용
    with tf.GradientTape() as tape:
        prediction = linear_regression(X)
        loss = mean_square_error(prediction, Y)
    
    # 손실 함수에 대한 경사 계산
    gradients = tape.gradient(loss, [W, b])
    
    # 경사 하강을 통해 변수 업데이트
    optimizer.apply_gradients(zip(gradients, [W, b]))
    
    if step % 100 == 0:
        print(step, loss.numpy(), W.numpy(), b.numpy())
#%%
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

tf.model = tf.keras.Sequential()
# units == output shape, input_dim == input shape
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

sgd = tf.keras.optimizers.SGD(lr=0.1)  # SGD == standard gradient descendent, lr == learning rate
tf.model.compile(loss='mse', optimizer=sgd)  # mse == mean_squared_error, 1/m * sig (y'-y)^2

# prints summary of the model to the terminal
tf.model.summary()

# fit() executes training
tf.model.fit(x_train, y_train, epochs=200)

# predict() returns predicted value
y_predict = tf.model.predict(np.array([5, 4]))
print(y_predict)