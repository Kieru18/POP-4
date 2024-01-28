import tensorflow as tf
from tensorflow import keras
from CONST import SEED
from neural_networks import FNN, CNN, CRNN
from utils import set_seed, plot_confusion_matrix, make_predictions, calculate_metrics

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


# #### TESTY DE ####################
# fnn_model = FNN()
# fnn_model.build(input_shape=(1, 28, 28))
# initial_weights = fnn_model.get_weights()

# print(type(initial_weights))

# weights_shapes = np.asarray([array.shape for array in initial_weights])
# flatten_weights = np.concatenate([w.flatten() for w in fnn_model.get_weights()])

# # print(weights_shapes)
# # print(weights_shapes.shape)
# # print(flatten_weights)
# # print(flatten_weights.shape)


# def reshape_weights(weights_shapes, flatten_weights):
#     reshaped_arrays = []
#     start_index = 0
#     for shape in weights_shapes:
#         size = np.prod(shape)
#         flattened_array = flatten_weights[start_index:start_index + size]
#         reshaped_array = flattened_array.reshape(shape)
#         reshaped_arrays.append(reshaped_array)
#         start_index += size

#     return reshaped_arrays


# reshaped = reshape_weights(weights_shapes, flatten_weights)
# print(reshaped)

# # for layer_weights in fnn_model.get_weights():
# #     print("Layer Weights:")
# #     print(layer_weights)
# #     print("Shape:")
# #     print(layer_weights.shape)
# #     print()


# def objective_function(flatten_weights, weights_shape, model, x_train, y_train):
#     weights = reshape_weights(weights_shape, flatten_weights)
#     model.set_weights(weights)
#     _, accuracy = model.evaluate(x_train, y_train, verbose=0)
#     return -accuracy  # Negate for maximization


# result = tfp.optimizer.differential_evolution_minimize(
#     lambda flatten_weights: objective_function(flatten_weights, weights_shapes, fnn_model, test_images, test_labels),
#     initial_position=initial_weights,
#     population_size=10,
#     max_iterations=5,
#     population_stddev=0.1,
#     seed=42
# )

# best_weights = result.position.numpy().reshape(-1)
# best_accuracy = -result.objective_value.numpy()  # Negate back to get accuracy







#### DEAP


def fitness_function(weights, model):
    print('___________________')
    print(f'Weights: {weights[0].shape}')
    model.set_weights(weights)
    predicted_labels = make_predictions(model, test_images)
    print('___________________')
    print(f'Labels P: {predicted_labels.shape} {predicted_labels}')
    print('___________________')
    print(f'Labels P: {np.argmax(test_labels, axis=1).shape} {np.argmax(test_labels, axis=1)}')

    _, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    # Załóżmy, że chcemy minimalizować sumę wag jako przykład
    print('___________________')
    print(f'Acc: {-accuracy}')
    return -accuracy


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def generate_individual(model):
    weights_shapes = np.asarray([array.shape for array in model.get_weights()])
    individual = [np.random.uniform(low=-1.0, high=1.0, size=shape) for shape in weights_shapes]
    return creator.Individual(individual)


def generate_population(population_size, model):
    return (generate_individual(model) for _ in range(population_size))


def deap_evaluate(individual):
    print('___________________')
    print(f'Model: {individual.model}')
    print(f'Type: {type(individual)}')
    return fitness_function(individual, individual.model),


def deap_evolve(population_size, generations, model):
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual, model=model)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selBest)

    # Zainicjuj populację i ewaluuj ją
    pop = toolbox.population(n=population_size)

    # Dodaj model jako dodatkowy atrybut do każdego osobnika
    for ind in pop:
        ind.model = model

    fitnesses = list(map(deap_evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        # print(ind.model)

    # Ewolucja przez określoną liczbę pokoleń
    for gen in range(generations):
        # Wybór rodziców
        parents = toolbox.select(pop, len(pop))

        # Skrzyżowanie i mutacja
        offspring = tools.cxTwoPoint(parents[0], parents[1])
        offspring = list(toolbox.mutate(child) for child in offspring)

        print('___________________')
        print(f'Offsprings:')
        for i, ind in enumerate(offspring):
            offspring[i] = ind[0]
            print(type(offspring[i]))
            print(offspring[i][0])

        # Ewaluacja potomstwa
        fitnesses = list(map(deap_evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
            print(ind.model)

        # Zastąpienie starej populacji potomstwem
        pop[:] = offspring

    # Zwróć najlepszego osobnika
    return tools.selBest(pop, 1)[0]


fnn_model = FNN()

population_size = 10
generations = 20

# Stwórz model FNN
fnn_model = FNN()
fnn_model.build(input_shape=(1, 28, 28))
fnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
initial_weights = fnn_model.get_weights()
weights_shapes = np.asarray([array.shape for array in fnn_model.get_weights()])
flatten_weights = np.concatenate([w.flatten() for w in fnn_model.get_weights()])


# Uruchom ewolucję
best_individual = deap_evolve(population_size, generations, fnn_model)

# Ustaw wagi modelu na najlepsze znalezione przez algorytm
fnn_model.set_weights(best_individual)

_, accuracy = fnn_model.evaluate(test_images, test_labels, verbose=0)
# Załóżmy, że chcemy minimalizować sumę wag jako przykład
print('___________________')
print(f'Acc: {-accuracy}')



# ################ KONIEC TESTÓW 
# print(f"Best Weights: {best_weights}")
# print(f"Best Accuracy: {best_accuracy}")

# # Use the best weights in your model
# best_model = FNN()
# best_model.set_weights(best_weights)


# predicted_labels = make_predictions(best_model, test_images)

# accuracy, precision, recall, conf_matrix = calculate_metrics(predicted_labels, np.argmax(test_labels, axis=1))

# print(f"FNN Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
# print("Confusion Matrix:")
# print(conf_matrix)

# # Plot the confusion matrix
# plot_confusion_matrix(conf_matrix, class_names)
