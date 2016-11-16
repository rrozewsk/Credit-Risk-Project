__author__ = 'ryanrozewski'
import numpy as np
import pandas as pd
from pandas import DataFrame
from Scheduler import Scheduler
#from parameters import xR,simNumber,trim_start,trim_end,t_step,x0Vas,start,referenceDate,fee
from MonteCarloSimulators.Vasicek.vasicekMCSim import MC_Vasicek_Sim
from Curves.Corporates.CorporateDaily import CorporateRates
from scipy.optimize import minimize
from Curves.Corporates import CorporateDaily
from datetime import date
#from Scheduler.Scheduler import Scheduler


class CouponBond(object):
    def __init__(self, fee, coupon, start, maturity, freq, referencedate, observationdate):
        self.fee = fee
        self.coupon=coupon
        self.start = start
        self.maturity = maturity
        self.freq= freq
        self.referencedate = referencedate
        self.observationdate = observationdate
        self.myScheduler = Scheduler.Scheduler()
        self.delay = self.myScheduler.extractDelay(freq=freq)
        self.getScheduleComplete()
        self.ntimes=len(self.datelist)
        self.referencedate = referencedate
        self.pvAvg=0.0
        self.cashFlows = DataFrame()
        self.cashFlowsAvg = []
        self.yieldIn = 0.0

    def getScheduleComplete(self):
        self.datelist = self.myScheduler.getSchedule(start=self.start,end=self.maturity,freq=self.freq,referencedate=self.referencedate)
        fullset = list(sorted(list(set(self.datelist)
                                   .union([self.referencedate])
                                   .union([self.start])
                                   .union([self.maturity])
                                   .union([self.observationdate])
                                   )))
        return fullset,self.datelist

    def setLibor(self,libor):
        #print(self.referencedate)
        self.libor = libor/libor.loc[self.referencedate]
        #print(self.libor)
        self.ntimes = np.shape(self.datelist)[0]
        self.ntrajectories = np.shape(self.libor)[1]
        self.ones = np.ones(shape=[self.ntrajectories])

    def getExposure(self, referencedate):
        if self.referencedate!=referencedate:
            self.referencedate=referencedate
            self.getScheduleComplete()
        deltaT= np.zeros(self.ntrajectories)
        for i in range(1,self.ntimes):
            deltaTrow = ((self.datelist[i]-self.datelist[i-1]).days/365)*self.ones
            deltaT = np.vstack ((deltaT,deltaTrow) )
        self.cashFlows= self.coupon*deltaT
        principal = self.ones
        self.cashFlows[self.ntimes-1,:] +=  principal
        if(self.datelist[0]<= self.start):
            self.cashFlows[self.start]=-self.fee
        self.cashFlowsAvg = self.cashFlows.mean(axis=1)
        pv = self.cashFlows*self.libor.loc[self.datelist]
        self.pv = pv.sum(axis=0)
        self.pvAvg = np.average(self.pv)
        return self.pv

    def getPV(self,referencedate):
        self.getExposure(referencedate=referencedate)
        return self.pv/self.libor.loc[self.observationdate]

    def getLiborAvg(self, yieldIn, datelist):
        self.yieldIn = yieldIn
        time0 = datelist[0]
        # this function is used to calculate exponential single parameter (r or lambda) Survival or Libor Functions
        Z = np.exp(-self.yieldIn * pd.DataFrame(np.tile([(x - time0).days / 365.0 for x in self.datelist], reps=[self.ntrajectories,1]).T,index=self.datelist))
        return Z

    def getYield(self,price):
        # Fit model to curve data
        yield0 = 0.05 * self.ones
        self.price = price
        self.yieldIn = self.fitModel2Curve(x=yield0)
        return self.yieldIn


    def fitModel2Curve(self, x ):
        # Minimization procedure to fit curve to model
        results = minimize(fun=self.fCurve, x0=x)
        return results.x

    def fCurve(self, x):
        # raw data error function
        calcCurve = self.getLiborAvg(x, self.datelist)
        thisPV = np.multiply(self.cashFlows,calcCurve).mean(axis=1).sum(axis=0)
        error = 1e4 * (self.price - thisPV) ** 2
        return error


#myrates=CorporateDaily.CorporateRates()



#coupon=.05


#myBond=CouponBond(fee=1,start=trim_start,maturity=trim_end,coupon=coupon,freq='3M',referencedate=referenceDate,observationdate=trim_start)
#fullist,datelist=myBond.getScheduleComplete()
#libor=MC_Vasicek_Sim(x=xR,simNumber=500,t_step=t_step,datelist=fullist)
#myBond.setLibor(libor.getLibor())
#print(myrates.getCorporateQData('AAA',datelist=datelist,R=.4))

## TEST from MP #####
#myScheduler = Scheduler.Scheduler()
#observationdate = minDay = date(2005,3,31) # Observation Date
#maxDay = date(2010,1,10)  # Last Date of the Portfolio
#start = date(2005, 1, 30)
#maturity = start+ myScheduler.extractDelay("1Y")
#print(maturity)
#referenceDate = date(2005, 6, 30)  # 6 months after trim_start
#simNumber = 5
#R = 0.4
#inArrears = True
#freq = '3M'
#t_step = 1.0/365

#simNumber=5
#coupon = 0.08
#fee= 1.0
#xR = [3.0, 0.05, 0.04, 0.03]


#myBond = CouponBond(fee=fee, start=start, maturity=maturity, coupon=coupon, freq="1M", referencedate=referenceDate, observationdate = observationdate)
#fulllist, datelist = myBond.getScheduleComplete()
#print(datelist)
#print(fulllist)
### Import Libor vasicek ##
#myMC = MC_Vasicek_Sim(x=xR, datelist=datelist, simNumber=simNumber, t_step=t_step)
#print(myMC.getLibor())
#libor = myMC.getSmallLibor(tenors=datelist)
#print("Get the Libor")
#print(libor)
## Import Q ###
#print("Corporate data ")
#myQ = CorporateRates()
#print("GEt Q")
#corporateQ = myQ.getCorporateQData(rating='AA',datelist=datelist,R=0.4)
### Create Z bar ##
#print(corporateQ['1 MO'])
#print(libor.iloc[:,1])
#Q1M = corporateQ
#Zbar = Q1M
#for i in range(Q1M.shape[1]):
#    Zbar.iloc[:,i] =Q1M.iloc[:,i]*libor.iloc[:,1]
#print("Zbar")
#print(Zbar)

## USe Bond Class ##
#myBond.setLibor(Zbar)

#print("Exposure")
#exposure = myBond.getExposure(referencedate=referenceDate)
#print("Present Value")
#pv = myBond.getPV(referencedate=referenceDate)
#print(pv[1])

## Test getYield ##
#yOpt = myBond.getYield(pv[1])
#print(yOpt)