from CONST import SEED
from utils import set_seed,  make_predictions
from sklearn.metrics import accuracy_score
import copy

import numpy as np
import json

from tensorflow import keras
from deap import base, creator, tools
from keras.losses import categorical_crossentropy

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


def get_accuracy(weights, model, test_images, test_labels):
    model.set_weights(weights)

    predicted_labels = make_predictions(model, test_images)
    accuracy = accuracy_score(np.argmax(test_labels, axis=1), predicted_labels)
    print('___________________')
    print(f'Acc: {-accuracy}')
    return -accuracy


def fitness_function(weights, model, test_images, test_labels, varaint='acc'):
    model.set_weights(weights)

    if varaint == 'acc':
        predicted_labels = make_predictions(model, test_images)
        accuracy = accuracy_score(np.argmax(test_labels, axis=1), predicted_labels)
        print('___________________')
        print(f'Acc: {-accuracy}')
        return -accuracy

    if varaint == 'cross':
        predicted_probs = model.predict(test_images)
        loss = categorical_crossentropy(test_labels, predicted_probs)
        print('___________________')
        print(f'Cross-entropy Loss: {np.sum(loss.numpy())}')
        return np.sum(loss.numpy())




def generate_individual(model):
    weights_shapes = np.asarray([array.shape for array in model.get_weights()], dtype=object)
    individual = [np.random.uniform(low=-1, high=1, size=shape) for shape in weights_shapes]
    return creator.Individual(individual)


def generate_population(population_size, model):
    return (generate_individual(model) for _ in range(population_size))


def deap_evaluate(individual, test_images, test_labels, varaint):
    return fitness_function(individual, individual.model, test_images, test_labels, varaint),


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


def de_best_1_bin_evolve(population_size, generations, model, F, CR, test_images, test_labels, variant, start_pop=None):
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual, model=model)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selBest)

    pop = toolbox.population(n=population_size)

    if start_pop:
        for i, ind in enumerate(pop):
            new = copy.deepcopy(np.asarray(start_pop[i], dtype=object))
            pop[i] = creator.Individual(new)

    for ind in pop:
        ind.model = model

    for i in range(population_size):
        pop[i].fitness.values = deap_evaluate(pop[i], test_images, test_labels, variant)

    history = []

    for gen in range(generations):
        print()
        print('--------------------------')
        print(f'Evolution - generation: {gen}')

        for i in range(len(pop)):
            print(f'Individual number: {gen}_{i}')
            wp = toolbox.select(pop, 1)[0]
            working_point = np.asarray(toolbox.select(pop, 1)[0], dtype=object)

            print(f'Fitness of working point: {deap_evaluate(wp, test_images, test_labels, variant)}')

            indexes = np.random.choice(range(len(pop)), 2, replace=False)

            point_d = np.asarray(pop[indexes[0]], dtype=object)
            point_e = np.asarray(pop[indexes[1]], dtype=object)

            point_M = working_point + F*(point_d - point_e)

            point_O = crossover(point_M, working_point, CR)

            point_O = creator.Individual(point_O)
            point_O.model = model
            point_O.fitness.values = deap_evaluate(point_O, test_images, test_labels, variant)
 
            if point_O.fitness.values < pop[i].fitness.values:
                print(f'old point fitness {pop[i].fitness.values}')
                print(f'new point fitness {point_O.fitness.values}')
                pop[i] = point_O

        history.append(get_accuracy(toolbox.select(pop, 1)[0], model, test_images, test_labels))

    for i in range(population_size):
        pop[i].fitness.values = deap_evaluate(pop[i], test_images, test_labels, variant)

    best = toolbox.select(pop, 1)[0]
    print(best.fitness.values)
    return best, pop, history


def mu_lambda_es_evolve(mu, lambda_, sigma, generations, test_images, test_labels, model,  variant, start_pop=None):
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual, model=model)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selBest)

    pop = toolbox.population(n=mu)

    if start_pop:
        for i, ind in enumerate(pop):
            new = copy.deepcopy(np.asarray(start_pop[i], dtype=object))
            pop[i] = creator.Individual(new)
        
    for ind in pop:
        ind.model = model

    for i in range(len(pop)):
        pop[i].fitness.values = deap_evaluate(pop[i], test_images, test_labels, variant)

    history = []

    for generation in range(generations):
        print()
        print('--------------------------')
        print(f'Evolution - generation: {generation}')
        offspring = []
        for i in range(mu):
            print(f'Individual number: {generation}_{i}')
            weights = np.asarray(pop[i], dtype=object)
            for i in range(lambda_):
                mutated_weights = [w + sigma * np.random.randn(*w.shape) for w in weights]
                mutated_ind = creator.Individual(mutated_weights)
                mutated_ind.model = model
                mutated_ind.fitness.values = deap_evaluate(mutated_ind, test_images, test_labels, variant)

                offspring.append(mutated_ind)

        combined_pop = pop + offspring

        pop = toolbox.select(combined_pop, mu)

        history.append(get_accuracy(toolbox.select(pop, 1)[0], model, test_images, test_labels))

    for i in range(len(pop)):
        pop[i].fitness.values = deap_evaluate(pop[i], test_images, test_labels, variant)

    best = toolbox.select(pop, 1)[0]
    print(best.fitness.values)
    return best, pop, history


def adapt_F(F_history):
    if len(F_history) >= 3:
        recent_changes = np.diff(F_history[-3:])
        if all(change > 0 for change in recent_changes):
            return F_history[-1] * 0.9
        elif all(change < 0 for change in recent_changes):
            return F_history[-1] * 1.1

    return F_history[-1]


def adapt_CR(CR_history):
    if len(CR_history) >= 3:
        recent_changes = np.diff(CR_history[-3:])
        if all(change > 0 for change in recent_changes):
            return CR_history[-1] * 0.9
        elif all(change < 0 for change in recent_changes):
            return CR_history[-1] * 1.1

    return CR_history[-1]


def jade_evolve(population_size, generations, model, F, CR, test_images, test_labels, variant, start_pop=None):
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual, model=model)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selBest)

    mu_F = 0.5
    mu_CR = 0.5
    c = 0.1
    
    pop = toolbox.population(n=population_size)

    if start_pop:
        for i, ind in enumerate(pop):
            new = copy.deepcopy(np.asarray(start_pop[i], dtype=object))
            pop[i] = creator.Individual(new)

    for ind in pop:
        ind.model = model

    for i in range(population_size):
        pop[i].fitness.values = deap_evaluate(pop[i], test_images, test_labels, variant)

    history = []

    for gen in range(generations):
        print()
        print('--------------------------')
        print(f'Ewolucja generacja: {gen}')
        
        F_values = np.random.standard_cauchy(population_size) + mu_F
        CR_values = np.random.normal(mu_CR, 0.1, population_size)

        successful_F = []
        successful_CR = []
        
        for i in range(len(pop)):
            F = F_values[i]
            CR = CR_values[i]
            print(f'Osobnik populacji numer: {i}')
            
            wp = toolbox.select(pop, 1)[0]
            working_point = np.asarray(toolbox.select(pop, 1)[0], dtype=object)

            print(f'Fitness of wp: {deap_evaluate(wp, test_images, test_labels, variant)}')

            indexes = np.random.choice(range(len(pop)), 2, replace=False)

            point_d = np.asarray(pop[indexes[0]], dtype=object)
            point_e = np.asarray(pop[indexes[1]], dtype=object)

            point_M = working_point + F*(point_d - point_e)

            point_O = crossover(point_M, working_point, CR)

            point_O = creator.Individual(point_O)
            point_O.model = model
            point_O.fitness.values = deap_evaluate(point_O, test_images, test_labels, variant)

            if point_O.fitness.values < pop[i].fitness.values:
                print(f'old point fitness {pop[i].fitness.values}')
                print(f'new point fitness {point_O.fitness.values}')
                pop[i] = point_O
                successful_F.append(F)
                successful_CR.append(CR)

        history.append(get_accuracy(toolbox.select(pop, 1)[0], model, test_images, test_labels))

        if successful_F:
            mL_SF = np.sum(np.square(successful_F)) / np.sum(successful_F)
            mu_F = (1 - c) * mu_F + c * mL_SF

        if successful_CR:
            mA_SCR = np.mean(successful_CR)
            mu_CR = (1 - c) * mu_CR + c * mA_SCR

    for i in range(population_size):
        pop[i].fitness.values = deap_evaluate(pop[i], test_images, test_labels, variant)

    best = toolbox.select(pop, 1)[0]
    print(best.fitness.values)
    return best, pop, history
