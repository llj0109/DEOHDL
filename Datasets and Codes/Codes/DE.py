# -*- coding: utf-8 -*-
"""
Created on Aug,2025

@author: Lujia Lv
"""
import argparse
import numpy as np
import random
import ensemble_boxes_wbf as wbf

def parse_args():
    parser = argparse.ArgumentParser(description="Population evolution configuration")
    parser.add_argument('--min_range', type=int, default=0,
                        help='The minimum range of values for each individual.')
    parser.add_argument('--max_range', type=int, default=1,
                        help='The maximum range of values for each individual.')
    parser.add_argument('--dim', type=int, default=3,
                        help='Dimension for each individual.')
    parser.add_argument('--rounds', type=int, default=40,
                        help='Rounds for population evolution.')
    parser.add_argument('--size', type=int, default=5,
                        help='Size for population.')
    parser.add_argument('--strategy_mu', type=str, default='DE/rand/1',
                        help='Mutational strategies in differential evolution.')
    parser.add_argument('--strategy_cr', type=str, default='arithmetic',
                        help='Crossover strategies in differential evolution.')
    return parser.parse_args()


class Population:
    def __init__(self, min_range, max_range, dim, rounds, size, object_func, CR=0.5, factor=0.8, strategy_mu='DE/rand/1', strategy_cr='arithmetic'):
        self.min_range = min_range
        self.max_range = max_range
        self.dimension = dim
        self.factor = factor
        self.rounds = rounds
        self.size = size
        self.cur_round = 0
        self.CR = CR
        self.get_object_function_value = object_func
        self.individuality = [np.array([random.uniform(self.min_range, self.max_range) for s in range(self.dimension)])
                              for tmp in range(size)]
        self.object_function_values = [self.get_object_function_value(v) for v in self.individuality]
        self.mutant = None
        self.cross = None
        self.strategy_mu = strategy_mu
        self.strategy_cr = strategy_cr

    def mutate(self):
        SFGSS = 8
        SFHC = 20
        Fl = 0.1
        Fu = 0.9
        tuo1 = 0.1
        tuo2 = 0.03
        tuo3 = 0.07

        rand1 = np.random.rand()
        rand2 = np.random.rand()
        rand3 = np.random.rand()
        F = np.random.rand()

        if rand3 < tuo2:
            F = SFGSS
        elif tuo2 <= rand3 < tuo3:
            F = SFHC
        elif rand2 < tuo1 and rand3 > tuo3:
            F = Fl + Fu * rand1

        self.mutant = []
        for i in range(self.size):
            tmp = None
            if self.strategy_mu == 'DE/rand/1':
                r1, r2, r3 = 0, 0, 0
                while r1 == r2 or r2 == r3 or r1 == r3 or r1 == i or r2 == i or r3 == i:
                    r1 = random.randint(0, self.size - 1)
                    r2 = random.randint(0, self.size - 1)
                    r3 = random.randint(0, self.size - 1)
                tmp = self.individuality[r1] + (self.individuality[r2] - self.individuality[r3]) * F

            if self.strategy_mu == 'DE/best/1':
                r1, r2 = 0, 0
                while r1 == r2 or r1 == i or r2 == i:
                    r1 = random.randint(0, self.size - 1)
                    r2 = random.randint(0, self.size - 1)

                self.object_function_values = np.array(self.object_function_values)
                best = np.argmax(self.object_function_values[:, 2])
                tmp = self.individuality[best] + (self.individuality[r1] - self.individuality[r2]) * F

            if self.strategy_mu == 'DE/rand/2':
                r1, r2, r3, r4, r5 = 0, 0, 0, 0, 0
                while r1 == r2 or r2 == r3 or r3 == r4 or r1 == r4 or r1 == r3 or r2 == r4 or r1 == r5 or r2 == r5 or r3 == r5 or r4 == r5 or r1 == i or r2 == i or r3 == i or r4 == i or r5 ==i:
                    r1 = random.randint(0, self.size - 1)
                    r2 = random.randint(0, self.size - 1)
                    r3 = random.randint(0, self.size - 1)
                    r4 = random.randint(0, self.size - 1)
                    r5 = random.randint(0, self.size - 1)

                tmp = self.individuality[r5] + (self.individuality[r1] - self.individuality[r2] + self.individuality[r3] - self.individuality[r4]) * F

            if self.strategy_mu == 'DE/best/2':
                r1, r2, r3, r4 = 0, 0, 0, 0
                while r1 == r2 or r2 == r3 or r3 == r4 or r1 == r4 or r1 == r3 or r2 == r4 or r1 == i or r2 == i or r3 == i or r4 == i:
                    r1 = random.randint(0, self.size - 1)
                    r2 = random.randint(0, self.size - 1)
                    r3 = random.randint(0, self.size - 1)
                    r4 = random.randint(0, self.size - 1)

                self.object_function_values = np.array(self.object_function_values)
                best = np.argmax(self.object_function_values[:, 2])
                tmp = self.individuality[best] + (self.individuality[r1] - self.individuality[r2] + self.individuality[r3] - self.individuality[r4]) * F

            if self.strategy_mu == 'DE/randtobest/1':
                r1, r2, r3 = 0, 0, 0
                while r1 == r2 or r2 == r3 or r1 == r3 or r1 == i or r2 == i or r3 == i:
                    r1 = random.randint(0, self.size - 1)
                    r2 = random.randint(0, self.size - 1)
                    r3 = random.randint(0, self.size - 1)

                self.object_function_values = np.array(self.object_function_values)
                best = np.argmax(self.object_function_values[:, 2])
                tmp = self.individuality[r1] + F * (self.individuality[best] - self.individuality[r1]) + F * (self.individuality[r2] - self.individuality[r3])

            for t in range(self.dimension):
                if tmp[t] > self.max_range:
                    tmp[t] = self.max_range
                if tmp[t] < self.min_range:
                    tmp[t] = self.min_range
            self.mutant.append(tmp)

    def crossover_bin(self):
        self.cross = []
        for i in range(self.size):
            temp_Ui = []
            Jrand = random.randint(0, self.dimension-1)
            for j in range(self.dimension):
                if random.random() < self.CR or j == Jrand:
                    U_j = self.mutant[i][j]
                else: U_j = self.individuality[i][j]
                temp_Ui.append(U_j)
            temp_Ui = np.array(temp_Ui)
            self.cross.append(temp_Ui)


    def crossover_ari(self):
        K = np.random.rand()
        self.cross = []
        for i in range(self.size):
            temp_Ui = self.individuality[i] + K * (self.mutant[i] - self.individuality[i])
            for t in range(self.dimension):
                if temp_Ui[t] > self.max_range:
                    temp_Ui[t] = self.max_range
                if temp_Ui[t] < self.min_range:
                    temp_Ui[t] = self.min_range
            self.cross.append(temp_Ui)

    def select(self):
        for i in range(self.size):
            tmp = self.get_object_function_value(self.cross[i])
            if tmp[2] > self.object_function_values[i][2]:
                self.individuality[i] = self.cross[i]
                self.object_function_values[i] = tmp

    def evolution(self):
        while self.cur_round < self.rounds:
            self.mutate()
            if self.strategy_cr == 'arithmetic':
                self.crossover_ari()
            if self.strategy_cr == 'binomial':
                self.crossover_bin()
            self.select()
            self.cur_round = self.cur_round + 1

def f(v):
    mp, mr, map50, map = wbf.run(v, dataset='valid')
    return mp, mr, map50, map

if __name__ == "__main__":
    args = parse_args()
    p = Population(min_range=args.min_range,
                   max_range=args.max_range,
                   dim=args.dim,
                   rounds=args.rounds,
                   size=args.size,
                   object_func=f,
                   strategy_mu=args.strategy_mu,
                   strategy_cr=args.strategy_cr)
    p.evolution()