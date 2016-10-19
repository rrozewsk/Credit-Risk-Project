__author__ = 'ryanrozewski'
import numpy as np
from pandas import DataFrame
from Scheduler import Scheduler
from parameters import xR,simNumber,trim_start,trim_end,t_step
from MonteCarloSimulators.Vasicek.vasicekMCSim import MC_Vasicek_Sim
from scipy.optimize import minimize




#every class should probably have a reference date if it is not in the future then problem
#each product has a start and reference date, should be flexible
#reference can be used to calculate EXPECTED POSITIVE EXPOSURE 
#or expected negative exposure. which is the PV of netting node in the future
class Bond(object):
    def __init__(self, libor, coupon, start,maturity,frequency,reference_date):
        self.libor=libor
        self.coupon=coupon
        schedule=Scheduler.Scheduler()
        self.delay=schedule.extractDelay(freq=frequency)
        self.datelist=schedule.getDatelist(start=start,end=maturity,freq=frequency,ref_date=reference_date)
        self.ntimes=len(self.datelist)
        self.pvAvg=0.0
        self.ntimes = np.shape(self.libor)[0]
        self.ntrajectories = np.shape(self.libor)[1]
        self.cashFlows = DataFrame()
        self.reference_date=reference_date
        return
    #this function is missing the usage of libor 
    #missing the discounting of cashflows plus principal to get PV
    #
    def PV(self):
        deltaT= np.zeros(self.ntrajectories)
        ones = np.ones(shape=[self.ntrajectories])
        for i in range(1,self.ntimes):
            deltaTrow = ((self.datelist[i]-self.datelist[i-1]).days/365)*ones
            deltaT = np.vstack ((deltaT,deltaTrow) )
        self.cashFlows= self.coupon*deltaT
        principal = ones
        self.cashFlows[self.ntimes-1,:] +=  principal
        pv = self.cashFlows*self.libor
        self.pvAvg = np.average(pv,axis=1)
        return self.pvAvg
    def getYield(self):
        pass
    
    def calculateFcurve(self,libor):
        #I think what we want to do here is tune the parameters of the interest rate model
        # So we want to create a method from within the montecarlo simulators to get the theta kappa and sigma
        # then we pass it to the calibrate and it gives us a min error and we compare the errors and choose the 
        # params with the least errors and then we can set the libor models params to these new found best tuned params
        pass
    
    def calibrateCurve(self,fcurve,x):
        fSum=np.sum(fcurve)
        xSum=np.sum(x)
        error=0
        error=minimize((fSum-xSum)**2)
        return error


myScheduler=Scheduler.Scheduler()
datelist=myScheduler.getSchedule(start=trim_start, end=trim_end, freq="3M")

coupon=.03
libor=MC_Vasicek_Sim(x=xR,simNumber=500,t_step=t_step,datelist=datelist)
myBond=Bond(start=trim_start,maturity=trim_end,coupon=coupon,frequency='3M',reference_date=trim_start,libor=libor.getSmallLibor(x=xR,tenors=datelist,simNumber=500))
print(myBond.PV())
