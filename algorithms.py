from CONST import SEED
from utils import set_seed,  make_predictions
from sklearn.metrics import accuracy_score
import copy

import numpy as np
import json 

from deap import base, creator, tools


set_seed(SEED)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

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


def fitness_function(weights, model, test_images, test_labels):
    model.set_weights(weights)
    predicted_labels = make_predictions(model, test_images)
    accuracy = accuracy_score(np.argmax(test_labels, axis=1), predicted_labels)
    print('___________________')
    print(f'Acc: {-accuracy}')
    return -accuracy


def generate_individual(model):
    weights_shapes = np.asarray([array.shape for array in model.get_weights()], dtype=object)
    individual = [np.random.uniform(low=-1, high=1, size=shape) for shape in weights_shapes]
    return creator.Individual(individual)


def generate_population(population_size, model):
    return (generate_individual(model) for _ in range(population_size))


def deap_evaluate(individual, test_images, test_labels):
    return fitness_function(individual, individual.model, test_images, test_labels),


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


def de_rand_1_bin_evolve(population_size, generations, model, F, CR, test_images, test_labels, start_pop=None):
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
            print(fitness_function(pop[i], model, test_images, test_labels))

    # Dodaj model jako dodatkowy atrybut do każdego osobnika
    for ind in pop:
        ind.model = model

    for i in range(population_size):
        pop[i].fitness.values = deap_evaluate(pop[i], test_images, test_labels)

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

            print(f'Fitness of wp: {deap_evaluate(wp, test_images, test_labels)}')

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
            point_O.fitness.values = deap_evaluate(point_O, test_images, test_labels)
            # Dodanie punktu do historii

            # Zamiana punktu, jeżeli p_O jest lepszy
            if point_O.fitness.values < pop[i].fitness.values:
                print(f'old point fitness {pop[i].fitness.values}')
                print(f'new point fitness {point_O.fitness.values}')
                pop[i] = point_O

        history.append(toolbox.select(pop, 1)[0].fitness.values)

    for i in range(population_size):
        pop[i].fitness.values = deap_evaluate(pop[i], test_images, test_labels)


    # Zwróć najlepszego osobnika
    best = toolbox.select(pop, 1)[0]
    print(best.fitness.values)
    return best, pop, history


def mu_lambda_es_evolve(mu, lambda_, sigma, generations, test_images, test_labels, model):
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual, model=model)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selBest)

    pop = toolbox.population(n=mu)

    # Dodaj model jako dodatkowy atrybut do każdego osobnika
    for ind in pop:
        ind.model = model

    for i in range(len(pop)):
        pop[i].fitness.values = deap_evaluate(pop[i], test_images, test_labels)

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
                mutated_ind.fitness.values = deap_evaluate(mutated_ind, test_images, test_labels)

                offspring.append(mutated_ind)

        # Połącz rodziców i potomków
        combined_pop = pop + offspring

        # Wybierz μ najlepszych osobników
        pop = toolbox.select(combined_pop, mu)

        # Dodaj do historii
        history.append(toolbox.select(pop, 1)[0].fitness.values)

    for i in range(len(pop)):
        pop[i].fitness.values = deap_evaluate(pop[i], test_images, test_labels)

    # Zwróć najlepszego osobnika
    best = toolbox.select(pop, 1)[0]
    print(best.fitness.values)
    return best, pop, history