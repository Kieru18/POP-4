import tensorflow as tf
from tensorflow.keras import layers, models

class FNN(tf.keras.Model):
    def __init__(self):
        super(FNN, self).__init__()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(10, activation='softmax')

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.maxpool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.maxpool2 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(10, activation='softmax')

class RNN(tf.keras.Model):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = layers.LSTM(128, return_sequences=False, input_shape=(None, 28 * 28))
        self.fc = layers.Dense(10, activation='softmax')

class CRNN(tf.keras.Model):
    def __init__(self):
        super(CRNN, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.maxpool1 = layers.MaxPooling2D((2, 2))
        self.rnn = layers.LSTM(128, return_sequences=False)
        self.fc = layers.Dense(10, activation='softmax')
