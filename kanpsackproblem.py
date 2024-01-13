from knapsack import Knapsack01Problem
from deap import base, tools, creator, algorithms
import random, time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

seed = str(time.time()).replace(".", "")[8:]
seed = 11888291 # int(seed)
random.seed(seed)

#This callback function returns the value of the knapsack
def knapsackValue(individual):
  #The comma is important because we want to return an interable object
  return knapsack.getValue(individual),

# Genetic Algorithm constants:
POPULATION_SIZE = 60  #Number of chromosome in the population
P_CROSSOVER = 0.9  #probability for crossover
P_MUTATION = 0.1   #probability for mutating an individual
MAX_GENERATIONS = 60 #Max iteration
HALL_OF_FAME_SIZE = 1 #Number of best element to show


knapsack = Knapsack01Problem()
toolbox = base.Toolbox()
#I register a zeroOrOne Callback function that is based on randint function with arguments 0 or 1.
toolbox.register("zeroOrOne", random.randint, 0, 1)
#Crate a FitnessMax Class that inherit from Fitness class. Since the KnapSack is a max problem, the weights (that is an argument)is 1. 
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#Create a Individual Class that inherit from list class and has a fitness (with type FitnessMax) attibute
creator.create("Individual", list, fitness=creator.FitnessMax)
#I register a callback function that creates an individual with the function initRepeat that takes argument: Individual class, the zeroOrOne function and the lenght of the chromosomes 
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(knapsack))
#I register a callback function that creates the population with the function initRepeat that takes argument: list class and individual creator function.
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
#Register evaluation function that is the knapsackValue
toolbox.register("evaluate", knapsackValue)
#Register selection function. I choose tournament selection between two invidividuals
toolbox.register("select", tools.selTournament, tournsize=3)
#Register crossover function. I choose the two point crossover function
toolbox.register("mate", tools.cxTwoPoint)
#Register mutation function. I choose flipbit. Every individual has 1/(knpasack lenght) possibilities to be choosen. Most than one gene can be mutated.
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/len(knapsack))
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

#Create the population
population = toolbox.populationCreator(n=POPULATION_SIZE)
#Create a statistical object for algorithm evaluation
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)
#Start the genetic process 
population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER,
                                          mutpb=P_MUTATION, ngen=MAX_GENERATIONS, stats=stats,
                                          halloffame=hof, verbose=True)

#Take the best chromosome
best = hof.items[0]
print("Best individual ever ", best)
print("Best fitness ever ", best.fitness.values[0])
print("-- Knapsack Items = ")
knapsack.printItems(best)

#extract statistics 
maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

# plot statistics:
sns.set_style("whitegrid")
plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Max / Average Fitness')
plt.title('Max and Average fitness over Generations')
plt.show()
print(seed)