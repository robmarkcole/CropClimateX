import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import wandb
import numpy as np
import os
import math

class BasisEvolutionAlgo():
    """handles all the things like logging, init, etc. which is repeated for every algorithm
    """
    _early_stop_gen = 0
    _best_record = None
    
    def __init__(self, toolbox, n_pop, n_gen, cxpb, mutpb, stats=None,
                 log_freq=1, early_stop=0, monitor=None, online_log=False, seed=42,
                 log_stats=None, save_best=3, score_names=[], multiobjective=False,
                 noise_range=10, noise_decay=0, log_dir='results/', verbose=True) -> None:
        self.toolbox = toolbox
        self.n_gen = n_gen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.log_freq = log_freq
        self.early_stop = early_stop
        self.monitor = monitor
        self.online_log = online_log
        self.score_names = score_names
        self.noise_decay = noise_decay
        self.init_noise_range = noise_range
        self.verbose = verbose
        self.log_dir = log_dir

        # init seed + pop
        random.seed(seed)
        np.random.seed(seed)
        self.population = self.toolbox.population(n=n_pop) # create pop
        
        # init the hall of fame
        self._hof = tools.HallOfFame(save_best)
        # init the stats
        if multiobjective and len(score_names) > 1:
            vars = {f'{i+1}_{k}': tools.Statistics(lambda ind: ind.fitness.values[i]) for i,k in enumerate(score_names)}
            vars['0_fitness'] = tools.Statistics(key=lambda ind: ind.fitness.values)
            self.stats = tools.MultiStatistics(**vars)
        else:
            self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        if log_stats:
            for k,v in log_stats.items():
                self.stats.register(k, v)

        # init the logbook
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + (self.stats.fields if self.stats else [])
        if isinstance(self.stats, tools.MultiStatistics):
            for chap in self.stats.fields if stats else []:
                for v in self.stats.values():
                    if isinstance(v, tools.support.Statistics):
                        self.logbook.chapters[chap].header = v.fields

    def eval_pop(self, pop):
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        return len(invalid_ind)
    
    def initial_log(self, nevals):
        record = self.stats.compile(self.population) if self.stats else {}
        self.logbook.record(gen=0, nevals=nevals, **record)
        if self.online_log:
            wandb.log({'gen':0,'nevals':nevals,**record})
        if self.verbose:
            print(self.logbook.stream)

    def log(self, i, nevals, noise):
        record = self.stats.compile(self.population) if self.stats else {}
        self.logbook.record(gen=i, nevals=nevals, **record)
        if self.online_log:
            wandb.log({'gen':i,'nevals':nevals,'noise':noise,**record})
        if self.verbose:
            print(self.logbook.__str__(i))
        if (self.log_freq > 0 and i % self.log_freq == 0) or self.n_gen == i:
            # unpad the patches
            self.toolbox.log_image([self._hof[0]], i, is_best=True)
            # self.toolbox.log_image(self._hof[1:], i)
        return record

    def is_best(self, record, update=False):
        if self._best_record is None:
            self._best_record = record
            return True

        if self.stats and self.monitor:
            if isinstance(self.monitor, str):
                better = self._best_record[self.monitor] > record[self.monitor]
            elif len(self.monitor) < 2:
                better = self._best_record[self.monitor[0]] > record[self.monitor[0]]
            else:
                better = self._best_record[self.monitor[0]][self.monitor[1]] > record[self.monitor[0]][self.monitor[1]]
            # save the best individual
            if better and update:
                self._best_record = record
        return better

    def early_stopping(self, record):    
        if self.early_stop:
            if self.is_best(record, update=True):
                self._early_stop_gen = 0
            else:
                self._early_stop_gen += 1

            if self._early_stop_gen >= self.early_stop:
                print(f'Early stopping after {self._early_stop_gen} generations of no improvement in {self.monitor}')
                return True
        return False
    
    def apply_noise_decay(self, i):
        if self.noise_decay > 0:
            nr = np.exp(-i/self.noise_decay) * self.init_noise_range
            return math.ceil(nr)
        else:
            return self.init_noise_range

class EaSimple(BasisEvolutionAlgo):
    def call(self):
        # Evaluate the individuals with an invalid fitness
        nevals = self.eval_pop(self.population)

        if self._hof is not None:
            self._hof.update(self.population)

        self.initial_log(nevals)

        # Begin the generational process
        for gen in range(1, self.n_gen + 1):
            # Select the next generation individuals
            offspring = self.toolbox.select(self.population, k=len(self.population))

            # Vary the pool of individuals
            noise = self.apply_noise_decay(gen)
            offspring = varAnd_mod(offspring, self.toolbox, self.cxpb, self.mutpb, noise)

            # Evaluate the individuals with an invalid fitness
            nevals = self.eval_pop(offspring)

            # Update the hall of fame with the generated individuals
            if self._hof is not None:
                self._hof.update(offspring)

            # Replace the current population by the offspring
            self.population[:] = offspring

            record = self.log(gen, nevals, noise)

            if self.early_stopping(record):
                break
        
def varAnd_mod(population, toolbox, cxpb, mutpb, noise_range):

    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i], noise_range=noise_range)
            del offspring[i].fitness.values

    return offspring