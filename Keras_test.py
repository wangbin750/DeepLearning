# coding=utf-8

import pandas as pd
import numpy as np

#模型选择，Sequential-线性
from keras.models import Sequential
model = Sequential()

#构建网络各层
from keras.layers import Dense, Activation
model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
#配置网络
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#优化网络，SGD-随机梯度下降法
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

#训练数据
X1 = np.random.randn(100)
X2 = np.random.randn(100)*2
Y = 2*X1+X2+np.random.randn(100)*0.01
X_train = np.array([X1, X2]).T
Y_train = np.array(Y)
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)

X3 = np.random.randn(50)
X4 = np.random.randn(50)*2
Y2 = 2*X1+X2+np.random.randn(50)*0.01
X_test = np.array([X3, X4]).T
Y_test = np.array(Y2)

loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
#classes = model.predict_classes(X_test, batch_size=32)
#proba = model.predict_proba(X_test, batch_size=32)
