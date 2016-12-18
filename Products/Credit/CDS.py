__author__ = 'ryan rozewski'

import Scheduler.Scheduler as schedule
from MonteCarloSimulators.Vasicek.vasicekMCSim import MC_Vasicek_Sim
from Curves.Corporates.CorporateDaily import CorporateRates
from Products.Rates import CouponBond
from Scheduler.Scheduler import Scheduler
from datetime import date
from parameters import xR, t_step, simNumber, Coupon, fee
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd


######################## CDS class ######################################
## Gets Q and Z based on the reference list
## Calculates Zbar(t,t_i) = Q(t,t_i)Z(t,t_i) for t_i in referencedates
## Employs the CouponBond to calculate the present value of the cashflows

######### Potential improvements CDS class ##############################
## 1) CouponBond
##      - Only supports end-of-the-month reference dates and observationdates
##        This is due to the use of getSchedule in Scheduler. Additionally
##        observationdate is defined to be one of the couponsdates. This
##        may be imporved to let observationdate be any date before and after
##        start_date. In order to change this the generators of Q and Z
##        needs to be adjusted.

class CDS(object):
    def __init__(self, start_date, end_date, freq, coupon, referenceDate, rating, R=.4):
        ### Parameters passed to external functions
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.R = R
        self.notional = 1
        self.fee = fee
        self.rating = rating
        self.coupon = coupon
        self.referenceDate = referenceDate
        self.observationDate = referenceDate

        ## Get datelist ##
        self.myScheduler = Scheduler()
        # ReferenceDateList = self.myScheduler.getSchedule(start=referenceDate,end=end_date,freq=freq, referencedate=referenceDate)

        ## Delay start and maturity
        delay = self.myScheduler.extractDelay(self.freq)
        cashFlowDays = self.myScheduler.extractDelay("3M")
        ## Delay maturity and start date
        # self.start_date = self.start_date +SixMonthDelay
        # self.maturity =self.start_date + delay
        self.maturity = self.start_date+delay

        fulllist, datelist = self.getScheduleComplete()
        self.datelist = datelist
        self.portfolioScheduleOfCF = fulllist

        self.myZ = None
        self.myQ = None

    # ////////////////////////////////////////////////////////////////////////////////////////////#
    # ////////////////////////////////////////////////////////////////////////////////////////////#
    # ////////////////////////////// SET FUNCTIONS ///////////////////////////////////////////////#
    # ////////////////////////////////////////////////////////////////////////////////////////////#
    # ////////////////////////////////////////////////////////////////////////////////////////////#

    def setZ(self, Zin):
        ## Zin needs to be a pandas dataframe
        self.myZ = Zin

    def setQ(self, Qin):
        self.myQ = Qin

    # ////////////////////////////////////////////////////////////////////////////////////////////#
    # ////////////////////////////////////////////////////////////////////////////////////////////#
    # ////////////////////////////// GET FUNCTIONS ///////////////////////////////////////////////#
    # ////////////////////////////////////////////////////////////////////////////////////////////#
    # ////////////////////////////////////////////////////////////////////////////////////////////#

    # //////////////// Get Vasicek simulated Z(t,t_j)
    def getZ_Vasicek(self):
        ### Get Z(t,t_i) for t_i in datelist #####

        ## Simulate Vasicek model with paramters given in workspace.parameters
        # xR = [5.0, 0.05, 0.01, 0.05]
        # kappa = x[0]
        # theta = x[1]
        # sigma = x[2]
        # r0 = x[3]
        vasicekMC = MC_Vasicek_Sim(datelist=[self.referenceDate, self.maturity], x=xR, simNumber=100, t_step=t_step)
        self.myZ = vasicekMC.getLibor()

        self.myLibor = self.myZ
        self.myZ = self.myZ.loc[:, 0]

        ## get Libor Z for reference dates ##
        # self.myZ = vasicekMC.getSmallLibor(datelist= self.portfolioScheduleOfCF).loc[:,0]

    # //////////////// Get Corporates simulated Q(t,t_j)
    def getQ_Corporate(self):
        ## Use CorporateDaily to get Q for referencedates ##
        # print("GET Q")
        # print(self.portfolioScheduleOfCF)
        myQ = CorporateRates()
        myQ.getCorporatesFred(trim_start=self.referenceDate, trim_end=self.end_date)
        ## Return the calculated Q(t,t_i) for bonds ranging over maturities for a given rating
        daterange = pd.date_range(start=self.referenceDate, end=self.maturity).date
        self.myQ = myQ.getCorporateQData(rating=self.rating, datelist=daterange, R=self.R)

        return (self.myQ)

    # //////////////// Get Premium leg Z(t_i)( Q(t_(i-1)) + Q(t_i) )
    def getPremiumLegZ(self):
        ## Check if Z and Q exists
        if self.myZ is None:
            self.getZ_Vasicek()

        if self.myQ is None:
            self.getQ_Corporate()

        ## Choose 1month Q
        '''
        Q1M = self.myQ[self.freq]
        # Q1M = self.myQ["QTranche"]
        timed = self.portfolioScheduleOfCF[self.portfolioScheduleOfCF.index(self.referenceDate):]
        Q1M = Q1M.loc[timed]
        Q1M = Q1M.cumprod(axis=0)
        zbarPremLeg = self.myZ / self.myZ.loc[self.referenceDate]
        zbarPremLeg = zbarPremLeg.loc[timed]
        ## Calculate Q(t_i) + Q(t_(i-1))
        out = 0
        for i in range(1, len(Q1M)):
            out = out + (Q1M[(i - 1)] + Q1M[i]) * float((timed[i] - timed[i - 1]).days / 365) * zbarPremLeg[i]

        ## Calculate the PV of the premium leg using the bond class

        # zbarPremLeg = zbarPremLeg.cumsum(axis=0)
        zbarPremLeg = pd.DataFrame(zbarPremLeg, index=timed)
        # print("Premium LEg")
        PVpremiumLeg = out * (1 / 2)
        # print(PVpremiumLeg)
        ## Get coupon bond ###
        return PVpremiumLeg
        '''
        Q1M = self.myQ[self.freq]
        # Q1M = self.myQ["QTranche"]
        timed = self.portfolioScheduleOfCF[self.portfolioScheduleOfCF.index(self.referenceDate):]
        Q1M = Q1M.loc[timed]
        Q1M=Q1M.cumprod()
        zbarPremLeg = self.myZ / self.myZ.loc[self.referenceDate]
        zbarPremLeg = zbarPremLeg.loc[timed]
        ## Calculate Q(t_i) + Q(t_(i-1))
        Qplus = []
        out = 0
        for i in range(1, len(Q1M)):
            out = out + (Q1M[(i - 1)] + Q1M[i]) * float((timed[i] - timed[i - 1]).days / 365) * zbarPremLeg[i]
        ## Calculate the PV of the premium leg using the bond class
        # zbarPremLeg = zbarPremLeg.cumsum(axis=0)
        zbarPremLeg = pd.DataFrame(zbarPremLeg, index=timed)
        # print("Premium LEg")
        PVpremiumLeg = out * (1 / 2)
        # print(PVpremiumLeg)
        ## Get coupon bond ###
        return PVpremiumLeg

    # //////////////// Get Protection leg Z(t_i)( Q(t_(i-1)) - Q(t_i) )
    def getProtectionLeg(self):
        if self.myZ is None:
            self.getZ_Vasicek()

        if self.myQ is None:
            self.getQ_Corporate()

        # Q1M = self.myQ["QTranche"]



        ## Calculate Q(t_i) + Q(t_(i-1))
        # Qminus = np.gradient(np.array(Q1M))
        # print(Qminus)
        # QArray = np.array(Q1M)
        # QArray = np.insert(QArray, obj = 0, values = 1)
        # print(QArray)


        # Q1M = self.myQ["QTranche"]
        '''
        Q1M = self.myQ[self.freq]
        Q1M = Q1M.cumprod()
        timed = Q1M.index.tolist()
        timed = self.portfolioScheduleOfCF[self.portfolioScheduleOfCF.index(self.referenceDate):]
        Q1M = Q1M.loc[timed]
        zbarPremLeg = self.myZ / self.myZ.loc[self.referenceDate]
        zbarPremLeg = zbarPremLeg.loc[timed]

        ## Calculate Q(t_i) + Q(t_(i-1))
        Qplus = []
        out = 0
        for i in range(1, len(Q1M)):
            out = out + (Q1M[(i-1)] - Q1M[(i)]) * float((timed[i] - timed[i - 1]).days / 365) * zbarPremLeg[i]

        return(out)
        ## Calculate Z Bar ##
        
        Qminus = np.gradient(Q1M)

        zbarProtectionLeg = self.myZ / self.myZ.loc[self.referenceDate]
        for i in range(1,zbarProtectionLeg.shape[0]):
            zbarProtectionLeg.iloc[i] = -Qminus[i] * zbarProtectionLeg.iloc[i] * (1/365)

        ## Calculate the PV of the premium leg using the bond class

        zbarProtectionLeg = zbarProtectionLeg.cumsum(axis=0)
        zbarProtectionLeg = pd.DataFrame(zbarProtectionLeg, index=Q1M.index)
        PVprotectionLeg = (1 - self.R) * zbarProtectionLeg
        ## Get coupon bond ###
        return PVprotectionLeg.loc[self.maturity]
        '''
        Q1M = self.myQ[self.freq]
        Q1M = Q1M.cumprod()
        Qminus = np.gradient(Q1M)
        zbarProtectionLeg = self.myZ / self.myZ.loc[self.referenceDate]
        for i in range(zbarProtectionLeg.shape[0]):
            zbarProtectionLeg.iloc[i] = -Qminus[i] * zbarProtectionLeg.iloc[i] * (1 / 365)
        ## Calculate the PV of the premium leg using the bond class
        zbarProtectionLeg = zbarProtectionLeg.cumsum(axis=0)
        zbarProtectionLeg = pd.DataFrame(zbarProtectionLeg, index=Q1M.index)
        PVprotectionLeg = (1 - self.R) * zbarProtectionLeg
        ## Get coupon bond ###
        return PVprotectionLeg.loc[self.maturity]

    # /////////////////////// Functions to get the exposure sum [delta Z(t,t_j)( Q(t,t_(j-1)) +/- Q(t,t_j) )
    # ////// These are copied from Coupon bond
    def getScheduleComplete(self):
        self.datelist = self.myScheduler.getSchedule(start=self.start_date, end=self.maturity, freq='3M',
                                                     referencedate=self.referenceDate)
        self.ntimes = len(self.datelist)
        fullset = list(sorted(list(set(self.datelist)
                                   .union([self.referenceDate])
                                   .union([self.start_date])
                                   .union([self.maturity])
                                   .union([self.observationDate])
                                   )))
        return fullset, self.datelist

    # /////// Get exposure
    def getExposure(self, referencedate, libor):
        self.ntrajectories = np.shape(libor)[1]
        self.ones = np.ones(shape=[self.ntrajectories])
        deltaT = np.zeros(self.ntrajectories)
        if self.ntimes == 0:
            pdzeros = pd.DataFrame(data=np.zeros([1, self.ntrajectories]), index=[referencedate])
            self.pv = pdzeros
            self.pvAvg = 0.0
            self.cashFlows = pdzeros
            self.cashFlowsAvg = 0.0
            return self.pv
        for i in range(1, self.ntimes):
            deltaTrow = ((self.datelist[i] - self.datelist[i - 1]).days / 365) * self.ones
            deltaT = np.vstack((deltaT, deltaTrow))
        self.cashFlows = self.coupon * deltaT
        principal = self.ones
        if self.ntimes > 1:
            self.cashFlows[-1:] += principal
        else:
            self.cashFlows = self.cashFlows + principal
        if (self.datelist[0] <= self.start_date):
            self.cashFlows[0] = -self.fee * self.ones

        if self.ntimes > 1:
            self.cashFlowsAvg = self.cashFlows.mean(axis=1) * self.notional
        else:
            self.cashFlowsAvg = self.cashFlows.mean() * self.notional
        pv = self.cashFlows * libor.loc[self.datelist]
        self.pv = pv.sum(axis=0) * self.notional
        self.pvAvg = np.average(self.pv) * self.notional
        return self.pv

    # ///// Get the calculated Market to Market based on spread
    def getValue(self, spread=1, R=0, buyer=True):
        ## Assume V(t) = S/2 sum Z(t_i) (Q(t_i) + Q(t_{i-1})) - (1-R)sum Z(t_i) (Q(t_{i-1}) - Q(t_{i}))
        ## Premium leg = S/2 sum Z(t_i) (Q(t_i) + Q(t_{i-1}))
        ## Protection leg =sum Z(t_i) (Q(t_i) + Q(t_{i-1}))
        premiumLeg = self.getPremiumLegZ()
        protectionLeg = self.getProtectionLeg()

        mtm = (spread / 2) * premiumLeg - (1 - R) * protectionLeg
        if buyer is True:
            return mtm
        else:
            return -mtm

    def getSpread(self):

        out=self.getProtectionLeg()/self.getPremiumLegZ()

        return out.values[0]

    def changeGuessForSpread(self, x):
        '''
        inputs a x list of guesses for the Vasicek simulator than changes the myQ
        '''
        vasicekMC = MC_Vasicek_Sim(datelist=[self.referenceDate, self.maturity], x=x, simNumber=20, t_step=t_step)
        self.myQ = vasicekMC.getLibor()[0]
        self.myQ = pd.DataFrame(self.myQ, index=self.myQ.index)
        self.myQ.columns = [self.freq]
        spread = self.bootProtec() / self.bootPrem()
        return spread.values[0]

    def bootProtec(self):
        Q1M = self.myQ[self.freq]
        Q1M = Q1M.cumprod()
        Qminus = np.gradient(Q1M)
        zbarProtectionLeg = self.myZ / self.myZ.loc[self.referenceDate]
        for i in range(zbarProtectionLeg.shape[0]):
            zbarProtectionLeg.iloc[i] = -Qminus[i] * zbarProtectionLeg.iloc[i] * (1 / 365)
        ## Calculate the PV of the premium leg using the bond class
        zbarProtectionLeg = zbarProtectionLeg.cumsum(axis=0)
        zbarProtectionLeg = pd.DataFrame(zbarProtectionLeg, index=Q1M.index)
        PVprotectionLeg = (1 - self.R) * zbarProtectionLeg
        ## Get coupon bond ###
        return PVprotectionLeg.loc[self.maturity]

    def bootPrem(self):
        Q1M = self.myQ[self.freq]
        # Q1M = self.myQ["QTranche"]
        timed = self.portfolioScheduleOfCF[self.portfolioScheduleOfCF.index(self.referenceDate):]
        Q1M = Q1M.loc[timed]
        Q1M=Q1M.cumprod()
        zbarPremLeg = self.myZ / self.myZ.loc[self.referenceDate]
        zbarPremLeg = zbarPremLeg.loc[timed]
        ## Calculate Q(t_i) + Q(t_(i-1))
        Qplus = []
        out = 0
        for i in range(1, len(Q1M)):
            out = out + (Q1M[(i - 1)] + Q1M[i]) * float((timed[i] - timed[i - 1]).days / 365) * zbarPremLeg[i]
        ## Calculate the PV of the premium leg using the bond class
        # zbarPremLeg = zbarPremLeg.cumsum(axis=0)
        zbarPremLeg = pd.DataFrame(zbarPremLeg, index=timed)
        # print("Premium LEg")
        PVpremiumLeg = out * (1 / 2)
        # print(PVpremiumLeg)
        ## Get coupon bond ###
        return PVpremiumLeg

#### TEST FUNCTIONS ###

# Parameters
# t_step = 1.0 / 365.0
# simNumber = 10
# start_date = date(2006,2,28)
# end_date = date(2008,9,30)  # Last Date of the Portfolio
# start = date(2006, 2, 28)
# referenceDate = date(2006, 3, 10)

# Testing of functions
# testCDS = CDS(start_date = start_date,end_date=end_date,freq='6M',coupon=1,referenceDate=referenceDate,rating="CCC",R=.4)


# getPremLeg = testCDS.getPremiumLegZ()
# getProtectionLeg = testCDS.getProtectionLeg()
# print(getPremLeg)
# print(getProtectionLeg)
# print(testCDS.getSpread())
