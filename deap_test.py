import tensorflow as tf
from tensorflow import keras
from CONST import SEED
from neural_networks import FNN, CNN, CRNN
from utils import set_seed, plot_confusion_matrix, make_predictions, calculate_metrics
from sklearn.metrics import accuracy_score
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

def save_population(population, filename):
    serialized_population = []
    for ind in population:
        serialized_individual = [array.tolist() for array in ind]
        serialized_population.append(serialized_individual)

    with open(filename, 'w') as file:
        json.dump(serialized_population, file)


def load_population(filename):
    with open(filename, 'r') as file:
        serialized_population = json.load(file)

    population = []
    for serialized_individual in serialized_population:
        individual = [np.array(array) for array in serialized_individual]
        population.append(creator.Individual(individual))

    return population


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
    individual = [np.random.uniform(low=-1, high=1, size=shape) for shape in weights_shapes]
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
                    output[i1[0]][i2] = layer_n_child
                # for i3, value in enumerate(layer_n_child):
                #     if np.random.rand() > CR:
                #         output[i1[0]][i2][i3] = value
        else:
            for i2, value in enumerate(layer_child):
                if np.random.rand() > CR:
                    output[i1[0]][i2] = value
            
    return output


def deap_evolve(population_size, generations, model, F, CR, start_pop=None):
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual, model=model)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selBest)

    # Zainicjuj populację i ewaluuj ją
    pop = toolbox.population(n=population_size)
    # pop[0] = creator.Individual(copy.deepcopy(initial_weights))
    
    if start_pop:
        for i, ind in enumerate(pop):
            print(type(ind))
            new = copy.deepcopy(np.asarray(start_pop[i], dtype=object))
            pop[i] = creator.Individual(new)
            print(fitness_function(pop[i], model))

    # Dodaj model jako dodatkowy atrybut do każdego osobnika
    for ind in pop:
        ind.model = model

    fitnesses = list(map(deap_evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # dodaj punkty początkowe do historii
    history = []

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

            # Zamiana punktu, jeżeli p_O jest lepszy
            if point_O.fitness.values < pop[i].fitness.values:
                print(f'old point fitness {pop[i].fitness.values}')
                print(f'new point fitness {point_O.fitness.values}')
                pop[i] = point_O

        history.append(toolbox.select(pop, 1)[0].fitness.values)

    fitnesses = list(map(deap_evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Zwróć najlepszego osobnika
    best = toolbox.select(pop, 1)[0]
    print(best.fitness.values)
    return best, pop, history


def mu_lambda_es_evolve(mu, lambda_, sigma, generations, model):
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual, model=model)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selBest)

    pop = toolbox.population(n=mu)

    # Dodaj model jako dodatkowy atrybut do każdego osobnika
    for ind in pop:
        ind.model = model

    fitnesses = list(map(deap_evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # dodaj punkty początkowe do historii
    history = []

    # Ewolucja przez określoną liczbę pokoleń
    for generation in range(generations):
        print()
        print('--------------------------')
        print(f'Ewolucja generacja: {generation}')
        # Mutate the current weights
        offspring = []
        for i in range(mu):
            print(f'Osobnik populacji numer: {i}')
            weights = np.asarray(pop[i], dtype=object)
            for i in range(lambda_):
                mutated_weights = [w + sigma * np.random.randn(*w.shape) for w in weights]
                mutated_ind = creator.Individual(mutated_weights)
                mutated_ind.model = model
                mutated_ind.fitness.values = deap_evaluate(mutated_ind)

                offspring.append(mutated_ind)

        # Połącz rodziców i potomków
        combined_pop = pop + offspring

        # Wybierz μ najlepszych osobników
        pop = toolbox.select(combined_pop, mu)

        # Dodaj do historii
        history.append(toolbox.select(pop, 1)[0].fitness.values)

    fitnesses = list(map(deap_evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Zwróć najlepszego osobnika
    best = toolbox.select(pop, 1)[0]
    print(best.fitness.values)
    return best, pop, history


fnn_model = FNN()

population_size = 20
generations = 30

# Stwórz model FNN
# fnn_model = FNN()
# fnn_model.build(input_shape=(1, 28, 28))
fnn_model = tf.keras.models.load_model("models/fnn_model_gradient")

initial_weights = fnn_model.get_weights()
weights_shapes = np.asarray([array.shape for array in fnn_model.get_weights()], dtype=object)
flatten_weights = np.concatenate([w.flatten() for w in fnn_model.get_weights()])

# Uruchom ewolucję
start_pop = load_population('population_1.json')

# best_individual, pop, history = deap_evolve(population_size, generations, fnn_model, 1, 0.5, start_pop)
best_individual, pop, history = mu_lambda_es_evolve(mu=5, lambda_=3, sigma=0.2, generations=25, model=fnn_model)


save_population(pop, 'population_3.json')

with open('history_3.json', 'w') as file:
    print(type(history))
    json.dump(history, file)


# Ustawienie wag modelu na najlepsze znalezione przez algorytm
model = FNN()
model.build((1, 28, 28))
model.set_weights(best_individual)
predicted_labels = make_predictions(model, test_images)
accuracy = accuracy_score(np.argmax(test_labels, axis=1), predicted_labels)

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
