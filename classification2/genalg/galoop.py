from deap import base
from deap import creator
from deap import tools

import random
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import elitism
from hyperparameters import LOWER_BOUNDS, UPPER_BOUNDS
from problem import GeneticSearch
from classification2.training import create_dataset

NUM_OF_PARAMS = len(LOWER_BOUNDS)


class GAloop:
    def __init__(self, pop_size=20, p_co=0.9, p_mut=0.5, max_gen=10, hof=3, crowding_fa=20.0):
        self.POPULATION_SIZE = pop_size
        self.P_CROSSOVER = p_co
        self.P_MUTATION = p_mut
        self.MAX_GENERATIONS = max_gen
        self.HALL_OF_FAME_SIZE = hof
        self.CROWDING_FACTOR = crowding_fa

    def run(self,
            train_control_path: str,
            train_sugar_path: str,
            train_ammonia_path: str,
            val_control_path: str,
            val_sugar_path: str,
            val_ammonia_path: str,
            length,
            batch_size
            ):

        train_set = create_dataset(
            train_control_path,
            train_sugar_path,
            train_ammonia_path,
            length,
            batch_size
        )
        val_set = create_dataset(
            val_control_path,
            val_sugar_path,
            val_ammonia_path,
            length,
            batch_size
        )
        test = GeneticSearch(train_set, val_set)

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
    opt = parse()
    ga = GAloop(opt['pop_size'], opt['prob_cx'], opt['prob_mut'],
                opt['max_gen'], opt['hof'], opt['crwoding'])
    ga.run(opt['train_set_path'],
           opt['train_l_path'],
           opt['test_set_path'],
           opt['test_l_path'])
