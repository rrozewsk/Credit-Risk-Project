import numpy as np
import pandas as pd
from scipy.optimize import minimize
from parameters import trim_start,trim_end,referenceDate,x0Vas

from Products.Credit.CDS import CDS

from parameters import freq
from MonteCarloSimulators.Vasicek.vasicekMCSim import MC_Vasicek_Sim



class BootstrapperCDSLadder(object):
    #  Class with Bootstrapping methods
    #  It can be used with CDS Ladder or KK Ratings CDS Ladder Converted Values
    def __init__(self, start, freq, LiborFunc, QFunc, OISFunc,R,CDSList):
        self.start=start
        self.freq=freq
        self.libor=LiborFunc
        self.Q=QFunc
        self.R=R
        self.OIS=OISFunc
        self.listCDS=CDSList

    # %% GetSpread
    def getSpreadBootstrapped(self, xQ, myCDS, s_quotes):
        calcCurve=myCDS.changeGuessForSpread(xQ)
        error=1e4*(s_quotes-calcCurve[0])**2
        return error

    def getScheduleComplete(self):
        self.datelist = self.myScheduler.getSchedule(start=self.start,end=self.maturity,freq=self.freq,referencedate=self.referencedate)
        fullset = list(sorted(list(set(self.datelist)
                                   .union([self.referencedate])
                                   .union([self.start])
                                   .union([self.maturity])
                                   .union([self.observationdate])
                                   )))
        return fullset,self.datelist

    def getSpreadList(self, xQ):
        spread = {}
        orderedCDS=[]
        for i in range(0,len(self.freq)):
            for j in range(0,len(self.listCDS)):
                if(self.freq[i] == self.listCDS[j].freq):
                    orderedCDS.append(self.listCDS[j])
        for i in range(0,len(orderedCDS)):
                quotes=orderedCDS[i].getSpread()
                #print(quotes)
                spread[orderedCDS[i].freq]=self.CalibrateCurve(x0=xQ,quotes=quotes[0],myCDS=orderedCDS[i])[0:4]
        return spread

    #  Fit CDS Ladder using Vasicek,CRI,etc Model.  Input parameters are x0
    #  QFunCIR  with the name of the Q Model Function
    def CalibrateCurve(self, x0, quotes,myCDS):
        # Bootstrap CDS Ladder Directly
        results = minimize(self.getSpreadBootstrapped, x0, args=(myCDS, quotes),method='Nelder-Mead')
        return results.x
lad=BootstrapperCDSLadder(start=trim_start,freq=['3M','6M'],LiborFunc=None,QFunc=None,OISFunc=None,R=.4,CDSList=[CDS(start_date = trim_start,end_date=trim_end,freq="3M",coupon=1,referenceDate=referenceDate,rating="CCC",R=0)])
vas=MC_Vasicek_Sim(x=lad.getSpreadList(x0Vas)['3M'],datelist=lad.listCDS[0].portfolioScheduleOfCF,t_step=1/365,simNumber=200)
print(vas.getLibor()[0])