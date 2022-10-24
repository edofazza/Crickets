from deap import base
from deap import creator
from deap import tools

import random
import numpy as np
from problem import GeneticSearch

import matplotlib.pyplot as plt

import elitism
from variables import LOWER_BOUNDS, UPPER_BOUNDS

NUM_OF_PARAMS = len(LOWER_BOUNDS)

# Genetic Algorithm constants:
POPULATION_SIZE = 2
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5   # probability for mutating an individual
MAX_GENERATIONS = 2
HALL_OF_FAME_SIZE = 1
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# create the classifier loss test class: #TODO
train_set = np.load('trainset.npy')
train_labels = np.load('traininglabels.npy')
val_set = np.load('valset.npy')
val_labels = np.load('vallabels.npy')
test = GeneticSearch(train_set, train_labels, val_set, val_labels)

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
    return -test.get_val_accuracy(individual),

toolbox.register("evaluate", classificationLoss)


toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate",
                 tools.cxSimulatedBinaryBounded,
                 low=LOWER_BOUNDS,
                 up=UPPER_BOUNDS,
                 eta=CROWDING_FACTOR)

toolbox.register("mutate",
                 tools.mutPolynomialBounded,
                 low=LOWER_BOUNDS,
                 up=UPPER_BOUNDS,
                 eta=CROWDING_FACTOR,
                 indpb=1.0 / NUM_OF_PARAMS)

# create initial population (generation 0):
population = toolbox.populationCreator(n=POPULATION_SIZE)

# prepare the statistics object:
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

# define the hall-of-fame object:
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

# perform the Genetic Algorithm flow with hof feature added:
population, logbook = elitism.eaSimpleWithElitism(population,
                                                  toolbox,
                                                  cxpb=P_CROSSOVER,
                                                  mutpb=P_MUTATION,
                                                  ngen=MAX_GENERATIONS,
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