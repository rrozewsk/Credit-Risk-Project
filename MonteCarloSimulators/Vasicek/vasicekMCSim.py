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
    # Define Vasicek perameters: kappa, theta. sigma, r0
    x = (2.3, .043, .055, .022)
    # Instantiate MonteCarlo class
    simulator = MC_Vasicek_Sim(datelist = datelist, x = x, 
                               simNumber = simNumber)
    # Get mean LIBOR values for the 500 simulations corresponding
    # to the first day of each month
    avgLibor = simulator.liborAvg
"""

from pandas import DataFrame
import numpy as np
import pandas as pd
from parameters import WORKING_DIR
from perameters import 
import os
import bisect
__author__ = 'ryanrozewski, michaelkovarik,'


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
        be set to 1 day.
    simNumber (int): The number of times the simulation is to execute.
    datelist (list): A list of strings that are date-formatted (e.g.
        '2016-10-17').
    datelistlong (list): A list of days between (and including) 
        min(datelist) and max(datelist). Each element is of type 
        datetime.date.
    ntimes (list):  The length of datelistlong.
    libor (pandas DataFrame): A (1 + ntimes, simNumber) shaped array 
        that  contains the simulated discount curves. The zeroth column 
        contains the mean curve. The type of each element is 
        numpy.float64. The row labels are dates corresponding to
        nodes in the simulation.
    smallLibor (pandas DataFrame): A matrix subset of the 
        libor array. But it only contains rows corresponding to the 
        dates in `datelist` instead of `datelistlong`.
    liborAvg (numpy ndarray): A vector containing the mean
        simulated libor values. It is also the zeroth column of 
        `libor`.
    """
    def __init__(self, datelist,x, simNumber,t_step):
        """
        Perameters
        ----------
        datelist (list): A list of strimgs that are date-formatted,
            e.g. '2012-04-16'.
        x (tuple): A 4-tuple containing the Vasicek SDE perameters:
            kappa, theta, sigma, r0.
        simNumber (int): The number of simulations that is to be 
            executed.
        """
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
        self.liborAvg=pd.DataFrame()
        
    def getLibor(self):
        """Executes the simulations and returns the simulated libor curves.

        Returns
        -------
        A large 2D pandoc DataFrame. Each column represents a simulated value of
        the libor curve at a given point in time. Each row corresponds to a
        date in `datelonglist`. The zeroth column contains the mean value of
        the simulated libor curves. The row labels are the elements of 
        datelonglist.
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
        self.liborAvg=np.average(self.libor,axis=1)
        self.libor=np.c_[self.liborAvg,self.libor]
        self.libor = pd.DataFrame(self.libor,index=self.datelistlong)
        return self.libor

    def getSmallLibor(self, x=[], tenors=[], simNumber=1):
        """ Returns a matrix of simulated Libor values corresponding to
        the dates of datelist.

        Perameters
        ----------
        x: Optional. Does nothing.
        tenors (list): Optional. A list of strings that are date
            formatted.
        simNumber: Optional. Does nothing.

        Returns
        -------
        A pandoc DataFrame. The values contained in the frame
        corresponds to the simulated libor interest rates. The rows
        corresponds to the entries in datelist, NOT datelistlong
        (as is the case with getLibor()).
        """
        #calculate indexes
        # Get Libor simulated Curves for specific tenors or for all tenors if no datelist is provided
        # calculate indexes
        if (len(self.libor) == 0):
            self.getLibor()
        self.smallLibor = self.libor.loc[tenors]
        return pd.DataFrame(self.smallLibor, index=tenors)

    def getLiborAvg(self):
        if(len(self.libor) == 0):
            self.getLibor()
            return self.libor[0]
        else:
            return self.libor[0]

    def fitPerams(discountCurves):
        """Finds the SDE perameters of best fit for a given discount curve

        Perameters
        ----------
        discountCurves : pandoc DataFrame
            A dataFrame consisting of sample discount curves. The columns each contain
            one discount curve. The rows are labeled by dates.

        Returns
        -------
        tuple
            A tuple containing the SDE perameters
        """
        def error(perams):
            simulator = MC_Vasicek_Sim(datelist = list(discountCurves.index), x = perams, 
                                       simNumber = 100, t_step = 1.0 / 365)
            simulatedCurve = simulator.getLiborAvg()
            return np.sum((simulatedCurve - discountCurve) ** 2) # sum of squares
        initValues = [0.000377701101971, 0.06807420742631265, 0.020205128906558, 0.002073084987793]
        return scipy.minimize(error, initValues)


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
