from pandas import DataFrame
import numpy as np
import pandas as pd
from datetime import date
import math
from dateutil.relativedelta import relativedelta
from random import shuffle
import random
import fractions

from Scheduler.Scheduler import Scheduler
from MonteCarloSimulators.Vasicek.vasicekMCSim import MC_Vasicek_Sim

class ExactFunc(object):
    def __init__(self, start, underlyings):

        myScheduler = Scheduler()
        myDelays = []
        freqs = ['3M', '6M', '1Y', '3Y']
        for i in range(0, len(freqs)):
            myDelays.append(myScheduler.extractDelay(freqs[i]))
        AAA = {}
        for i in range(0, len(freqs)):
            vas = MC_Vasicek_Sim(x=AAAx[freqs[i]], datelist=[start, myDelays[i] + start], t_step=1 / 365.,
                                 simNumber=500)
            AAA[freqs[i]] = vas.getLibor()[0].loc[myDelays[i] + start]

        BBB = {'3M': MC_Vasicek_Sim(x=BBBx[freqs[0]], datelist=[start, myDelays[0] + start], t_step=1 / 365.,
                                    simNumber=500).getLibor()[0].loc[myDelays[0] + start]}
        self.probs = {'AAA': AAA, 'BBB': BBB}
        self.underlyings = underlyings

    def f(self, k, j):
        '''
        The recursion relation for the homogenous portfolio
        takes in k: an int for numer of defaults
        and j: number of underlyings you want to consider in the calculation k cannnot be greater than j



        '''
        if (j == 0 and k == 0):
            return 1
        if (j == 0 and k > 0):
            return 0
        if (k == 0 and j > 0):
            return self.f(k, j - 1) * self.probs[self.underlyings[j][3]][self.underlyings[j][2]]
        else:
            return self.f(k, j - 1) * (self.probs[self.underlyings[j][3]][self.underlyings[j][2]]) + self.f(k - 1,
                                                                                                            j - 1) * (
                                                                                                     1 - self.probs[
                                                                                                         self.underlyings[
                                                                                                             j][3]][
                                                                                                         self.underlyings[
                                                                                                             j][2]])

    '''
    Helper functions

    '''

    def gcd(self, x, y):
        while y != 0:
            (x, y) = (y, x % y)
        return x

    def totalGCD(self):
        g = (1 - self.underlyings[0][4]) * self.underlyings[0][0]
        for i in range(1, len(self.underlyings)):
            g = self.gcd(g, ((1 - self.underlyings[i][4]) * self.underlyings[i][0]))
        return g

    def getLossVec(self):
        g = self.totalGCD()
        n = []
        for i in range(0, len(self.underlyings)):
            n.append(((1 - self.underlyings[i][4]) * self.underlyings[i][0]) / g)
        return n

    def fprime(self, k, j, vec):
        '''
        recursion relation for inhomogenous portfolio takes
        k an int representing number of defaulted credits
        j an int representing number of underlyings we wish to consider
        vec a list of length of underlyings with the underlyings Loss given default scaled by gcd so
        each entry is an int
        '''
        if (j == 0 and k == 0):
            return 1
        if (j == 0 and k > 0):
            return 0
        if (0 < k and vec[j] > k):
            return self.fprime(k, j - 1, vec) * self.probs[self.underlyings[j][3]][self.underlyings[j][2]]
        if (vec[j] <= k and k <= np.array(vec[0:j]).sum()):
            return self.fprime(k, j - 1, vec) * (
            self.probs[self.underlyings[j][3]][self.underlyings[j][2]]) + self.fprime(k - vec[j], j - 1, vec) * (
            1 - self.probs[self.underlyings[j][3]][self.underlyings[j][2]])
        else:
            return self.fprime(k, j - 1, vec) * self.probs[self.underlyings[j][3]][self.underlyings[j][2]]


    def getTrancheNumb(self, K):
        sum = np.array(self.getLossVec()).sum()
        losses = self.getLossVec()
        totalLoss = 0
        for i in range(0, len(losses)):
            totalLoss = totalLoss + losses[i] / sum
            if (totalLoss >= K):
                return i


    def threshold(self, K):
        sum = np.array(self.getLossVec()).sum()
        return math.floor(sum * K)