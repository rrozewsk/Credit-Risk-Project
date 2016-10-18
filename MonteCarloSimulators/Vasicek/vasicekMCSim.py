# -*- coding: utf-8 -*-
"""
Vasicek.vasicekMCSim
====================

This class contains a Monte Carlo simulator for calculating Libor
discount curves using the Vasicek model of interest rates:

dr(t) = kappa * (theta - r(t)) dt  + sigma * dW

Example
-------

    # First day of each month
    datelist = ['2014-01-01',
                '2014-02-01',
                '2014-03-01',
                '2014-04-01']
    # run 500 simulations
    simNumber = 500
    # Define Vasicek perameters
    x = (2.3, .043, .055, .022)
    # Instantiate MonteCarlo class
    simulator = MC_Vasicek_Sim(datelist = datelist, x = x, 
                               simNumber = simNumber)
    # Get mean LIBOR values for the 500 simulations for the first day
      of each month
    avgLibor = simulator.getLibor()[0]
"""

from pandas import DataFrame
import numpy as np
import pandas as pd
from parameters import WORKING_DIR
import os
import bisect
__author__ = 'ryanrozewski'


class MC_Vasicek_Sim(object):
    """ Monte Carlo simulator for interest rates under the Vasicek 
    model.

    Attributes
    ----------
    kappa (float): Vasicek perameter: 'speed of reversion'.
    theta (float): Vasicek perameter: 'long term mean level'.
    sigma (float): Vasicek perameter: 'volatility'
    r0 (float): Vasicek perameter: 'initial value'.
    t_step (float): The time difference between the 'steps' in the 
        simulation. Represents 'dt' in the Vasicek model. Should always
        be set to 1.
    simNumber (int): The number of times the simulation is to execute.
    datelist (list): A list of strings that are date-formatted (e.g.
        '2016-10-17').
    datelistlong (list): A list of days between (and including) 
        min(datelist) and max(datelist). Each element is of type 
        datetime.date.
    ntimes (list):  The length of datelistlong.
    libor (numpy.ndarray): A (1 + ntimes, simNumber) shaped array that 
        contains the simulated discount curves. The zeroth column 
        contains the mean curve. The type of each element is 
        numpy.float64.
    smallLibor (pandas.core.frame.DataFrame): A matrix subset of the 
        libor array. But it only contains rows corresponding to the 
        dates in `datelist` instead of `datelistlong`.
    """
    def __init__(self, datelist,x, simNumber,t_step = 1):
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
        """Executes the simulations and returns the simulated libor curves.

        Returns
        -------
        A large 2D numpy ndarray. Each column represents a simulated value of
        the libor curve at a given point in time. Each row corresponds to a
        date in `datelonglist`. The zeroth column contains the mean value of
        the simulated libor curves.
        """
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
        """ Returns a matrix of simulated Libor values corresponding to
        the dates of datelist.

        Arguments
        ---------
        datelist (list): Optional. A list of strings that are date
            formatted. Defaults to the object's `datelist` value.

        Returns
        -------
        A pandoc DataFrame. The values contained in the frame
        corresponds to the simulated libor interest rates. The rows
        corresponds to the entries in datelist, NOT datelistlong
        (as is the case with getLibor()).
        """
        #calculate indexes
        if(datelist is None):
            datelist=self.datelist
        ind = self.return_indices1_of_a(self.datelistlong, datelist)
        df=DataFrame(self.getLibor())
        self.smallLibor=df.loc[ind]
        return self.smallLibor

#####################################################################################
    def saveMeExcel(self):
        """ Saves the value of 'libor' as OpenXML spreadsheet.
        """
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

