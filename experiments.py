import tensorflow as tf
from CONST import SEED, BATCH_SIZE
from neural_networks import FNN, CNN, RNN, CRNN
from utils import set_seed, evaluate, plot_confusion_matrix, make_predictions, calculate_metrics

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


set_seed(SEED)


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

plt.imshow(train_images[0].reshape(28, 28), cmap='gray')
# plt.imshow(train_images[0][:, :, 0], cmap='gray')


# Example usage for FNN
class_names = [str(i) for i in range(10)]  # Assuming you have 10 classes for MNIST

# train model, and predict labels
fnn_model = FNN()
fnn_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
fnn_model.fit(train_images, train_labels, epochs=5, batch_size=64, shuffle=True)

predicted_labels = make_predictions(fnn_model, test_images)

accuracy, precision, recall, conf_matrix = calculate_metrics(predicted_labels, np.argmax(test_labels, axis=1))

print(f"FNN Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plot_confusion_matrix(conf_matrix, class_names)
