import tensorflow as tf
from tensorflow import keras
from CONST import SEED
from neural_networks import FNN, CNN, CRNN
from utils import set_seed, plot_confusion_matrix, make_predictions, calculate_metrics
from sklearn.metrics import accuracy_score
from algorithms import mu_lambda_es_evolve, de_rand_1_bin_evolve, load_population, save_population
import copy
import json

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow_probability as tfp
from keras.initializers import RandomUniform
from deap import base, creator, tools


set_seed(SEED)


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))

test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

plt.imshow(train_images[0].reshape(28, 28), cmap='gray')

class_names = [str(i) for i in range(10)]


fnn_model = FNN()

population_size = 20
generations = 1

# Stwórz model FNN
fnn_model = FNN()
fnn_model.build(input_shape=(1, 28, 28))
# fnn_model = tf.keras.models.load_model("models/fnn_model_gradient")

# Uruchom ewolucję
start_pop = load_population('population_1.json')

best_individual1, pop, history = de_rand_1_bin_evolve(population_size, generations, fnn_model, 1, 0.5, test_images=test_images,
                                                      test_labels=test_labels, start_pop=start_pop)

# best_individual2, pop, history = mu_lambda_es_evolve(mu=2, lambda_=2, sigma=0.2, generations=2, test_images=test_images,
#                                                     test_labels=test_labels, model=fnn_model)


# save_population(pop, 'population_4.json')

# with open('history_3.json', 'w') as file:
#     print(type(history))
#     json.dump(history, file)


# Ustawienie wag modelu na najlepsze znalezione przez algorytm
model = FNN()
model.build((1, 28, 28))

model.set_weights(best_individual1)
predicted_labels = make_predictions(model, test_images)
accuracy = accuracy_score(np.argmax(test_labels, axis=1), predicted_labels)

print('Model Output')
print(f'Accuracy: {accuracy}')

# model.set_weights(best_individual2)
# predicted_labels = make_predictions(model, test_images)
# accuracy = accuracy_score(np.argmax(test_labels, axis=1), predicted_labels)

# print('Model Output')
# print(f'Accuracy: {accuracy}')


predicted_labels = make_predictions(model, test_images)

accuracy, precision, recall, conf_matrix = calculate_metrics(predicted_labels, np.argmax(test_labels, axis=1))

print(f"FNN Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plot_confusion_matrix(conf_matrix, class_names, title="Confusion matrix FNN - DE")
