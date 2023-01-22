from deap import base
from deap import creator
from deap import tools

import tensorflow as tf

import random
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os

import elitism
from hyperparameters import LOWER_BOUNDS, UPPER_BOUNDS
from problem import GeneticSearch

NUM_OF_PARAMS = len(LOWER_BOUNDS)


"""def create_dataset(control_path, sugar_path):
    data, labels = [], []
    tmp_list = [c for c in os.listdir(control_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(control_path, i))
        data.append(tmp_npy)
        labels.append(0)

    tmp_list = [c for c in os.listdir(sugar_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(sugar_path, i))
        data.append(tmp_npy)
        labels.append(1)

    return normalize(np.array(data)), np.array(labels)"""
def divide_sequence(sequence, length):
    if length == 3480:
        return np.array(sequence)
    _, dim = sequence.shape
    tmp = []
    for i in range(dim - length):
        tmp_seq = sequence[:, i:i + length]
        tmp.append(tmp_seq)
    return np.array(tmp)


def create_dataset(control_path, sugar_path, ammonia_path=None, length=3480):
    if length < 3480:
        data = None
    else:
        data = []
    labels = []
    tmp_list = [c for c in os.listdir(control_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(control_path, i))
        labels.append(0)
        if type(data) is list:
            data.append(tmp_npy)
        else:
            if data is None:
                data = divide_sequence(tmp_npy, length)
            else:
                data = np.r_[data, divide_sequence(tmp_npy, length)]

    tmp_list = [c for c in os.listdir(sugar_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(sugar_path, i))
        labels.append(1)
        if type(data) is list:
            data.append(tmp_npy)
        else:
            data = np.r_[data, divide_sequence(tmp_npy, length)]

    if ammonia_path is not None:
        tmp_list = [c for c in os.listdir(ammonia_path) if c.endswith('.npy')]
        for i in tmp_list:
            tmp_npy = np.load(os.path.join(ammonia_path, i))
            labels.append(2)
            if type(data) is list:
                data.append(tmp_npy)
            else:
                data = np.r_[data, divide_sequence(tmp_npy, length)]

    if type(data) is list:
        return normalize(np.array(data)), np.array(labels)
    else:
        return normalize(data), np.array(labels)


def normalize(x):
    return tf.keras.utils.normalize(x, axis=-1)


class GAloop:
    def __init__(self, pop_size=250, p_co=0.9, p_mut=0.5, max_gen=50, hof=10, crowding_fa=20.0):
        self.POPULATION_SIZE = pop_size
        self.P_CROSSOVER = p_co
        self.P_MUTATION = p_mut
        self.MAX_GENERATIONS = max_gen
        self.HALL_OF_FAME_SIZE = hof
        self.CROWDING_FACTOR = crowding_fa

    def run(self,
            train_control_path: str,
            train_sugar_path: str,
            val_control_path: str,
            val_sugar_path: str
            ):

        random.seed(42)
        train_set, train_labels = create_dataset(
            train_control_path,
            train_sugar_path
        )
        val_set, val_labels = create_dataset(
            val_control_path,
            val_sugar_path
        )

        test = GeneticSearch(train_set, train_labels, val_set, val_labels, classes=3)

        toolbox = base.Toolbox()

        # define a single objective, maximizing fitness strategy:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # create the Individual class based on list:
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # define the hyperparameter attributes individually:
        for i in range(NUM_OF_PARAMS):
            # "hyperparameter_0", "hyperparameter_1", ...
            toolbox.register("hyperparameter_" + str(i),
                             random.uniform,
                             LOWER_BOUNDS[i],
                             UPPER_BOUNDS[i])

        # create a tuple containing an attribute generator for each param searched:
        hyperparameters = ()
        for i in range(NUM_OF_PARAMS):
            hyperparameters = hyperparameters + \
                              (toolbox.__getattribute__("hyperparameter_" + str(i)),)

        # create the individual operator to fill up an Individual instance:
        toolbox.register("individualCreator",
                         tools.initCycle,
                         creator.Individual,
                         hyperparameters,
                         n=1)

        # create the population operator to generate a list of individuals:
        toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

        # fitness calculation
        def classificationLoss(individual):
            return -test.get_val_loss(individual),

        toolbox.register("evaluate", classificationLoss)

        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("mate",
                         tools.cxSimulatedBinaryBounded,
                         low=LOWER_BOUNDS,
                         up=UPPER_BOUNDS,
                         eta=self.CROWDING_FACTOR)

        toolbox.register("mutate",
                         tools.mutPolynomialBounded,
                         low=LOWER_BOUNDS,
                         up=UPPER_BOUNDS,
                         eta=self.CROWDING_FACTOR,
                         indpb=1.0 / NUM_OF_PARAMS)

        # create initial population (generation 0):
        population = toolbox.populationCreator(n=self.POPULATION_SIZE)

        # prepare the statistics object:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        stats.register("avg", np.mean)

        # define the hall-of-fame object:
        hof = tools.HallOfFame(self.HALL_OF_FAME_SIZE)

        # perform the Genetic Algorithm flow with hof feature added:
        population, logbook = elitism.eaSimpleWithElitism(population,
                                                          toolbox,
                                                          cxpb=self.P_CROSSOVER,
                                                          mutpb=self.P_MUTATION,
                                                          ngen=self.MAX_GENERATIONS,
                                                          stats=stats,
                                                          halloffame=hof,
                                                          verbose=True)

        # print best solution found:
        print("- Best solution is: ")
        print("params = ", hof.items[0])
        print("Accuracy = %1.5f" % hof.items[0].fitness.values[0])
        np.save('hof.npy', hof.items)

        # extract statistics:
        maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")
        np.save('maxFit.npy', maxFitnessValues)
        np.save('avgFit.npy', meanFitnessValues)

        # plot statistics:
        plt.plot(maxFitnessValues, color='red')
        plt.plot(meanFitnessValues, color='green')
        plt.xlabel('Generation')
        plt.ylabel('Max / Average Fitness')
        plt.title('Max and Average fitness over Generations')
        plt.savefig('ga_search.png')


def parse():
    parser = ArgumentParser()
    parser.add_argument('--pop_size', type=int, default=5)
    parser.add_argument('--prob_cx', type=float, default=0.9)
    parser.add_argument('--prob_mut', type=float, default=0.5)
    parser.add_argument('--max_gen', type=int, default=2)
    parser.add_argument('--hof', type=int, default=1)
    parser.add_argument('--crowding', type=float, default=20.0)

    parser.add_argument('--train_set_path', type=str, default='train_set.npy')
    parser.add_argument('--train_l_path', type=str, default='train_l.npy')
    parser.add_argument('--test_set_path', type=str, default='test_set.npy')
    parser.add_argument('--test_l_path', type=str, default='test_l.npy')
    return vars(parser.parse_args())


if __name__ == '__main__':
    """opt = parse()
    ga = GAloop(opt['pop_size'], opt['prob_cx'], opt['prob_mut'],
                opt['max_gen'], opt['hof'], opt['crwoding'])
    ga.run(opt['train_set_path'],
           opt['train_l_path'],
           opt['test_set_path'],
           opt['test_l_path'])"""
    ga = GAloop()
    ga.run( # control-sugar control-ammonia sugar-ammonia
        'prediction_head_centered/control/train/',
        'prediction_head_centered/ammonia/train/',
        'prediction_head_centered/control/val/',
        'prediction_head_centered/ammonia/val/'
    )
