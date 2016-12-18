import numpy as np
import pandas as pd
from scipy.optimize import minimize
from parameters import trim_start,trim_end,referenceDate,x0Vas
from datetime import date
from Products.Credit.CDS import CDS
from parameters import freq
from MonteCarloSimulators.Vasicek.vasicekMCSim import MC_Vasicek_Sim

class BootstrapperCDSLadder(object):
    #  Class with Bootstrapping methods
    #  It can be used with CDS Ladder or KK Ratings CDS Ladder Converted Values
    def __init__(self, start, freq, R,CDSList):
        self.start=start
        self.freq=freq
        self.R=R
        self.listCDS=CDSList

    # %% GetSpread
    def getSpreadBootstrapped(self, xQ, myCDS, s_quotes):
        calcCurve=myCDS.changeGuessForSpread(xQ)
        error=(s_quotes-calcCurve)**2
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
                myQ=MC_Vasicek_Sim(x=self.CalibrateCurve(x0=xQ,quotes=quotes,myCDS=orderedCDS[i])[0:4],datelist=[orderedCDS[i].referenceDate,orderedCDS[i].maturity],t_step=1/365,simNumber=1000).getLibor()[0]
                myQ=pd.DataFrame(myQ.values,columns=[orderedCDS[i].freq],index=myQ.index)
                orderedCDS[i].myQ=myQ
                #print(myQ)
                spread[orderedCDS[i].freq]=orderedCDS[i].getSpread()
        return spread

    def getXList(self, xQ):
        out = {}
        orderedCDS=[]
        for i in range(0,len(self.freq)):
            for j in range(0,len(self.listCDS)):
                if(self.freq[i] == self.listCDS[j].freq):
                    orderedCDS.append(self.listCDS[j])
        for i in range(0,len(orderedCDS)):
                quotes=orderedCDS[i].getSpread()
                #print(quotes)
                out[orderedCDS[i].freq]=self.CalibrateCurve(x0=xQ,quotes=quotes,myCDS=orderedCDS[i]).tolist()
        return out

    #  Fit CDS Ladder using Vasicek,CRI,etc Model.  Input parameters are x0
    #  QFunCIR  with the name of the Q Model Function
    def CalibrateCurve(self, x0, quotes,myCDS):
        # Bootstrap CDS Ladder Directly
        results = minimize(self.getSpreadBootstrapped, x0, args=(myCDS, quotes),method='Powell')
        print(results.success)
        print(myCDS.freq)
        return results.x

'''
myLad=BootstrapperCDSLadder(start=trim_start,freq=['3M'],CDSList=[CDS(start_date=trim_start,end_date=date(2010,1,1),freq='3M',coupon=1,referenceDate=trim_start,rating='AAA')],R=.4).getSpreadList(x0Vas)
print(myLad)
'''