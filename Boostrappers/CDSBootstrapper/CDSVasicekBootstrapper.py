import numpy as np
import pandas as pd
from scipy.optimize import minimize


from Products.Credit.CDS import CDS
from parameters import freq



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
        calcCurve=myCDS.getSpread(xQ)
        cashFlows=myCDS.cashFlows
        price=np.multiply(cashFlows,calcCurve).mean(axis=1).sum(axis=0)
        error=1e4*(s_quotes-price)**2
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
                s_quote=0
                spread[orderedCDS[i].freq]=self.CalibrateCurve(x0=xQ,quotes=s_quote)
        return spread

    #  Fit CDS Ladder using Vasicek,CRI,etc Model.  Input parameters are x0
    #  QFunCIR  with the name of the Q Model Function
    def CalibrateCurve(self, x0, quotes,myCDS):
        # Bootstrap CDS Ladder Directly
        results = minimize(self.getSpreadBootstrapped, x0, (myCDS, quotes))
        return results.x
