__author__ = 'ryanrozewski'
from pandas import DataFrame
import numpy as np
import pandas as pd
from parameters import WORKING_DIR
import os
import bisect


class MC_Vasicek_Sim(object):
    def __init__(self, datelist,x, simNumber,t_step):
    #SDE parameters - Vasicek SDE
    # dr(t) = k(θ − r(t))dt + σdW(t)
        self.kappa = x[0]
        self.theta = x[1]
        self.sigma = x[2]
        self.r0 = x[3]
        self.simNumber = simNumber
        self.t_step = t_step
    #internal representation of times series - integer multiples of t_step
        self.datelist = datelist
    #creation of a fine grid for Monte Carlo integration
        #Create fine date grid for SDE integration
        minDay = min(datelist)
        maxDay = max(datelist)
        self.datelistlong = pd.date_range(minDay, maxDay).tolist()
        self.datelistlong = [x.date() for x in self.datelistlong]
        self.ntimes = len(self.datelistlong)
        self.libor=[]
        self.smallLibor = []

    def getLibor(self):
        rd = np.random.standard_normal((self.ntimes,self.simNumber))   # array of numbers for the number of samples
        r = np.zeros(np.shape(rd))
        nrows = np.shape(rd)[0]
        sigmaDT = self.sigma* np.sqrt(self.t_step)
    #calculate r(t)
        r[1,:] = self.r0+r[1,:]
        for i in np.arange(2,nrows):
            r[i,:] = r[i-1,:]+ self.kappa*(self.theta-r[i-1,:])*self.t_step + sigmaDT*rd[i,:]
    #calculate integral(r(s)ds)
        integralR = r.cumsum(axis=0)*self.t_step
    #calculate Libor
        self.libor = np.exp(-integralR)
        
        return self.libor

    def getSmallLibor(self,datelist=None):
        #calculate indexes
        if(datelist is None):
            datelist=self.datelist
        ind = self.return_indices1_of_a(self.datelistlong, datelist)
        df=DataFrame(self.getLibor())
        self.smallLibor=df.loc[ind]
        return self.smallLibor

#####################################################################################
    def saveMeExcel(self):
        df = DataFrame(self.libor)
        df.to_excel(os.path.join(WORKING_DIR,'MC_Vasicek_Sim.xlsx'), sheet_name='libor', index=False)

#####################################################################################
    def return_indices1_of_a(self, a, b):
        b_set = set(b)
        ind = [i for i, v in enumerate(a) if v in b_set]
        return ind
#####################################################################################
    def return_indices2_of_a(self, a, b):
        index=[]
        for item in a:
            index.append(bisect.bisect(b,item))
        return np.unique(index).tolist()

