from deap import base
from deap import creator
from deap import tools

import tensorflow as tf

import random
import numpy as np
import matplotlib.pyplot as plt
import os

import elitism
from hyperparameters import LOWER_BOUNDS, UPPER_BOUNDS
from problem import GeneticSearch

NUM_OF_PARAMS = len(LOWER_BOUNDS)


def divide_sequence(sequence, length):
    """
    From a sequence (joints_axis, frames) the function reduces the frame-length to
    sequences of shape (joints_axis, length) and store them in a new np.array that
    the function returns
    :param sequence: numpy sequence
    :param length: size of the window
    :return: np.array containing all subsequences obtained from sequence
    """
    if length == 3480:
        return np.array(sequence)
    _, dim = sequence.shape
    tmp = []
    for i in range(dim - length):
        tmp_seq = sequence[:, i:i + length]
        tmp.append(tmp_seq)
    return np.array(tmp)


def normalize(x):
    """
    Normalize values in a sequence x considering its last axis
    :param x: sequence
    :return: normalized sequence
    """
    return tf.keras.utils.normalize(x, axis=-1)


def create_dataset(first_stimulus_path, second_stimulus_path, third_stimulus_path=None, length=3480):
    """
    Create numpy dataset
    :param first_stimulus_path: path to the numpy directory containing the npy sequences for the first stimulus
    :param second_stimulus_path: path to the numpy directory containing the npy sequences for the second stimulus
    :param third_stimulus_path: path to the numpy directory containing the npy third for the first stimulus (if None it will not be considered, i.e., binary case)
    :param length: length of the (sub)sequences, in our case we use as default the total length (3480)
    :return: a tuple in the form (data, labels)
    """
    if length < 3480:
        data = None
    else:
        data = []
    labels = []
    tmp_list = [c for c in os.listdir(first_stimulus_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(first_stimulus_path, i))
        if type(data) is list:
            data.append(tmp_npy)
            labels.append(0)
        else:
            if data is None:
                data = divide_sequence(tmp_npy, length)
                labels = data.shape[0]*[0]
            else:
                tmp_data = divide_sequence(tmp_npy, length)
                data = np.r_[data, tmp_data]
                labels.extend(tmp_data.shape[0] * [0])

    tmp_list = [c for c in os.listdir(second_stimulus_path) if c.endswith('.npy')]
    for i in tmp_list:
        tmp_npy = np.load(os.path.join(second_stimulus_path, i))
        if type(data) is list:
            data.append(tmp_npy)
            labels.append(1)
        else:
            tmp_data = divide_sequence(tmp_npy, length)
            data = np.r_[data, tmp_data]
            labels.extend(tmp_data.shape[0]*[1])

    if third_stimulus_path is not None:
        tmp_list = [c for c in os.listdir(third_stimulus_path) if c.endswith('.npy')]
        for i in tmp_list:
            tmp_npy = np.load(os.path.join(third_stimulus_path, i))
            if type(data) is list:
                data.append(tmp_npy)
                labels.append(2)
            else:
                tmp_data = divide_sequence(tmp_npy, length)
                data = np.r_[data, tmp_data]
                labels.extend(tmp_data.shape[0] * [2])

    if type(data) is list:
        return normalize(np.array(data)), np.array(labels)
    else:
        return normalize(data), np.array(labels)


class GAloop:
    def __init__(self, pop_size=200, p_co=0.9, p_mut=0.5, max_gen=10, hof=8, crowding_fa=20.0, multi_gpus=False):
        self.POPULATION_SIZE = pop_size
        self.P_CROSSOVER = p_co
        self.P_MUTATION = p_mut
        self.MAX_GENERATIONS = max_gen
        self.HALL_OF_FAME_SIZE = hof
        self.CROWDING_FACTOR = crowding_fa
        self.multi_gpus = multi_gpus

    def run(self,
            train_control_path: str,
            train_sugar_path: str,
            val_control_path: str,
            val_sugar_path: str,
            train_ammonia_path=None,
            val_ammonia_path=None,
            length=3480,
            shape=(8, 3480),
            batch_size=16,
            epochs=1000
            ):

        random.seed(42)
        train_set, train_labels = create_dataset(
            train_control_path,
            train_sugar_path,
            train_ammonia_path,
            length
        )
        val_set, val_labels = create_dataset(
            val_control_path,
            val_sugar_path,
            val_ammonia_path,
            length
        )

        test = GeneticSearch(train_set, train_labels, val_set, val_labels, classes=3,
                             multi_gpus=self.multi_gpus, shape=shape, batch_size=batch_size, epochs=epochs)

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


if __name__ == '__main__':
    ga = GAloop()
    ga.run( # control-sugar control-ammonia sugar-ammonia
        train_control_path='prediction_head_centered/control/train/',
        train_sugar_path='prediction_head_centered/sugar/train/',
        val_control_path='prediction_head_centered/control/val/',
        val_sugar_path='prediction_head_centered/sugar/val/',
        train_ammonia_path='prediction_head_centered/ammonia/train/',
        val_ammonia_path='prediction_head_centered/ammonia/val/',
        length=3190,
        shape=(8, 3190),
        epochs=10000
    )
