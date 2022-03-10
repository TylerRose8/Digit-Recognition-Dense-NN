# Testing the dense NN for classification using the MNIST image data

## Author: Bojian Xu, bojianxu@ewu.edu


import sys

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import nn
from code_misc.utils import MyUtils

sys.path.append('..')

k = 10  # number of classes
d = 784  # number of features, excluding the bias feature
time_import = time.time()
# READ in data
df_X_train = pd.read_csv('MNIST/x_train.csv', header=None)
df_y_train = pd.read_csv('MNIST/y_train.csv', header=None)
df_X_test = pd.read_csv('MNIST/x_test.csv', header=None)
df_y_test = pd.read_csv('MNIST/y_test.csv', header=None)

# save in numpy arrays
X_train_raw = df_X_train.to_numpy()
y_train_raw = df_y_train.to_numpy()
X_test_raw = df_X_test.to_numpy()
y_test_raw = df_y_test.to_numpy()
print('Time to import data: %.2f' % (time.time() - time_import))

# get training set size
n_train = X_train_raw.shape[0]
n_test = X_test_raw.shape[0]

time_normalize = time.time()
# normalize all features to [0,1]
X_all = MyUtils.normalize_0_1(
    np.concatenate((X_train_raw, X_test_raw), axis=0))  # np.concatenate((X_train_raw, X_test_raw),axis=0)
X_train = X_all[:n_train]
X_test = X_all[n_train:]
print('Time to normalize data: %.2f' % (time.time() - time_normalize))

# convert each label into a 0-1 vector
y_train = np.zeros((n_train, k))
y_test = np.zeros((n_test, k))
for i in range(n_train):
    y_train[i, int(y_train_raw[i])] = 1.0
for i in range(n_test):
    y_test[i, int(y_test_raw[i])] = 1.0

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# print(y_test)


# print(y_train[-10:])


# build the network
nuts = nn.NeuralNetwork()

nuts.add_layer(d=d)  # input layer - 0

nuts.add_layer(d=10, act='logis')  # hidden layer - 1
nuts.add_layer(d=5, act='logis')  # hiddent layer - 2
# nuts.add_layer(d = 100, act = 'relu')  # hiddent layer - 3
# nuts.add_layer(d = 30, act = 'relu')  # hiddent layer - 4

nuts.add_layer(d=k, act='relu')  # output layer,    multi-class classification, #classes = k

errors = nuts.fit(X_train, y_train, eta=0.1, iterations=5, SGD=False, mini_batch_size=20)

#x = np.arange(len(errors))
#plt.plot(x, errors)
#plt.show()

print(nuts.error(X_train, y_train))
print(nuts.error(X_test, y_test))

preds = nuts.predict(X_test)

# print(preds[:100])
# print(y_test_raw[:100])
# print(np.sum(preds != y_test_raw))

misclassified = 0
classified = 0

for i in range(y_test.shape[0]):
    if preds[i] != y_test_raw[i]:
        misclassified += 1
    else:
        classified += 1
#         print('misclassified!!')
#     print('predicted as', preds[i])
#     print('label is', y_test_raw[i])
#     pixels = X_test_raw[i].reshape((28, 28))
#     plt.imshow(pixels, cmap='gray')
#     plt.show()

print(misclassified)
print(classified)
