import tensorflow as tf
from tensorflow.keras import layers, models


class FNN(models.Model):
    def __init__(self):
        super(FNN, self).__init__()
        self.flatten = layers.Flatten(input_shape=(28, 28))
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


class CNN(models.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


class RNN(models.Model):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm = layers.LSTM(128, input_shape=(28, 28))
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.lstm(x)
        x = self.dense1(x)
        return self.dense2(x)


class CRNN(models.Model):
    def __init__(self):
        super(CRNN, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.reshape = layers.Reshape((13, 13 * 32))
        self.lstm = layers.LSTM(128, activation='relu')
        self.dense = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.reshape(x)
        x = self.lstm(x)
        return self.dense(x)
