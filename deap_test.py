import tensorflow as tf
from tensorflow import keras
from CONST import SEED
from neural_networks import FNN, CNN, CRNN
from utils import set_seed, plot_confusion_matrix, make_predictions, calculate_metrics
from sklearn.metrics import accuracy_score
import copy

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
    model.set_weights(weights)
    predicted_labels = make_predictions(model, test_images)

    accuracy = accuracy_score(np.argmax(test_labels, axis=1), predicted_labels)

    print('___________________')
    print(f'Acc: {-accuracy}')
    return -accuracy


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def generate_individual(model):
    weights_shapes = np.asarray([array.shape for array in fnn_model.get_weights()], dtype=object)
    individual = [np.random.uniform(low=-0.5, high=0.5, size=shape) for shape in weights_shapes]
    return creator.Individual(individual)


def generate_population(population_size, model):
    return (generate_individual(model) for _ in range(population_size))


def deap_evaluate(individual):
    return fitness_function(individual, individual.model),


def crossover(child, parent, CR):
    output = copy.deepcopy(parent)
    for i1, layer_child in np.ndenumerate(child):
        if len(layer_child.shape) == 2:
            for i2, layer_n_child in enumerate(layer_child):
                for i3, value in enumerate(layer_n_child):
                    if np.random.rand() > CR:
                        output[i1[0]][i2][i3] = value
        else:
            for i2, value in enumerate(layer_child):
                if np.random.rand() > CR:
                    output[i1[0]][i2] = value
            
    return output


def deap_evolve(population_size, generations, model, F, CR):
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual, model=model)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selBest)

    # Zainicjuj populację i ewaluuj ją
    pop = toolbox.population(n=population_size)
    # pop[0] = creator.Individual(copy.deepcopy(initial_weights))

    # Dodaj model jako dodatkowy atrybut do każdego osobnika
    for ind in pop:
        ind.model = model

    fitnesses = list(map(deap_evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # dodaj punkty początkowe do historii
    history = []
    history.append(pop)

    # Ewolucja przez określoną liczbę pokoleń
    for gen in range(generations):
        print()
        print('--------------------------')
        print(f'Ewolucja generacja: {gen}')

        for i in range(len(pop)):
            print(f'Osobnik populacji numer: {i}')
            # wybieramy punkt roboczy - najlepszy z populacji
            wp = toolbox.select(pop, 1)[0]
            working_point = np.asarray(toolbox.select(pop, 1)[0], dtype=object)

            print(f'Fintess of wp: {deap_evaluate(wp)}')

            # wybieramy dwa losowe punkty jako baza modyfikacji
            indexes = np.random.choice(range(len(pop)), 2, replace=False)

            point_d = np.asarray(pop[indexes[0]], dtype=object)
            point_e = np.asarray(pop[indexes[1]], dtype=object)

            # Utworzenie nowego punktu
            point_M = working_point + F*(point_d - point_e)

            # Krzyżowanie punktów

            point_O = crossover(point_M, working_point, CR)

            point_O = creator.Individual(point_O)
            point_O.model = model
            point_O.fitness.values = deap_evaluate(point_O)
            # Dodanie punktu do historii
            history.append(point_O)


            # Zamiana punktu, jeżeli p_O jest lepszy
            if point_O.fitness.values < pop[i].fitness.values:
                print(type(pop))
                print(f'old point fitness {type(pop[i])}')
                print(f'old point fitness {pop[i].fitness.values}')
                print(f'new point fitness {point_O.fitness.values}')
                print(type(pop[i]))
                print(type(point_O))
                pop[i] = point_O
                print(f'old point fitness {type(pop[i])}')
                print(f'new point fitness {pop[i].fitness.values}')

    fitnesses = list(map(deap_evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Zwróć najlepszego osobnika
    best = toolbox.select(pop, 1)[0]
    print(best.fitness.values)
    return (best)


fnn_model = FNN()

population_size = 10
generations = 2

# Stwórz model FNN
# fnn_model = FNN()
# fnn_model.build(input_shape=(1, 28, 28))
fnn_model = tf.keras.models.load_model("models/fnn_model_gradient")

initial_weights = fnn_model.get_weights()
weights_shapes = np.asarray([array.shape for array in fnn_model.get_weights()], dtype=object)
flatten_weights = np.concatenate([w.flatten() for w in fnn_model.get_weights()])

# for j, layer_child in np.ndenumerate(initial_weights):
#     print(type(layer_child))
#     print(layer_child.shape)
#     if len(layer_child.shape) > 1:
#         for v, layer_n_child in enumerate(layer_child):
#             print(type(layer_n_child))
#             for r, value in enumerate(layer_n_child):
#                 #swap values 
#                 value_right = 1 #value_left
#     else:
#         for value in layer_n_child:
#             #print(type(value))
#             #swap values 
#             value_right = 1 #value_left
        
#     # for k in range(layer_child.shape[0]):
#     #     if np.random.rand() > 0.5:
#     #         xaa = 2



# Uruchom ewolucję
best_individual = deap_evolve(population_size, generations, fnn_model, 2, 0.2)

# Ustaw wagi modelu na najlepsze znalezione przez algorytm
model = FNN()
model.build((1, 28, 28))
model.set_weights(best_individual)
predicted_labels = make_predictions(model, test_images)
accuracy = accuracy_score(np.argmax(test_labels, axis=1), predicted_labels)

# Załóżmy, że chcemy minimalizować sumę wag jako przykład
print('Model Output')
print(f'Accuracy: {accuracy}')



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
