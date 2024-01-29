from typing import Union

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Flatten, LSTM, Conv2D, MaxPooling2D, Reshape


class FNN(Model):
    def __init__(self, info: str = '', optimizer: Union[str, str] = 'sgd'):
        super(FNN, self).__init__()
        self.flatten = Flatten(input_shape=(28, 28))
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

        self.info = info
        self.optimizer = optimizer

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

    def __str__(self):
        return f'FNN_{self.info}'


class CNN(Model):
    def __init__(self, info: str = '', optimizer: Union[str, str] = 'sgd'):
        super(CNN, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.pool1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.pool2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

        self.info = info
        self.optimizer = optimizer

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

    def __str__(self):
        return f'CNN_{self.info}'



class CRNN(Model):
    def __init__(self, info: str = '', optimizer: Union[str, str] = 'sgd'):
        super(CRNN, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.pool1 = MaxPooling2D((2, 2))
        self.reshape = Reshape((13, 13 * 32))
        self.lstm = LSTM(128, activation='relu')
        self.dense = Dense(10, activation='softmax')

        self.info = info
        self.optimizer = optimizer

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.reshape(x)
        x = self.lstm(x)
        return self.dense(x)

    def __str__(self):
        return f'CRNN_{self.info}'
