# November 05 Hack sesh
# =====================
#
# Today, we should make sure the following work:
#   + [ ] Create a Monte Carlo simulation for a discount curve Z(t),
#         using realistic perameters (by fitting with actual Libor 
#         data).
#   + [ ] For any company/rating, create a credit curve Q(t) given Z(t)
#         and at least one of the following methods:
#           - [ ] Bootstrapping from a bond yield curve (use the credit
#                 triangle). This requires inputting a recovery rate R.
#           - [ ] Bootstrapping from CDS data. This is mostly insensive
#                 to changes in R (so set R = 0.4 or something).
#            -[ ] Derive Q(t) from equity curve (using Merton's Model or
#                 equivalent).
#   + [ ] Make a script that allows the user to create a portfolio consisting
#         of CDS's and Bonds and apply exposure analysis.

#matplotlib inline
from datetime import date
import time
import pandas as pd
import numpy as np
pd.options.display.max_colwidth = 60
from Curves.Corporates.CorporateDailyVasicek import CorporateRates
from Boostrappers.CDSBootstrapper.CDSVasicekBootstrapper import BootstrapperCDSLadder
from MonteCarloSimulators.Vasicek.vasicekMCSim import MC_Vasicek_Sim
from Products.Rates.CouponBond import CouponBond
from Products.Credit.CDS import CDS
from Scheduler.Scheduler import Scheduler
import quandl
import matplotlib.pyplot as plt
from parameters import WORKING_DIR
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*'))
from IPython.core.pylabtools import figsize
figsize(15, 4)
from pandas import ExcelWriter
import numpy.random as nprnd
from pprint import pprint

t_step = 1.0 / 365.0
simNumber = 10
trim_start = date(2005,3,10)
trim_end = date(2010,12,31)  # Last Date of the Portfolio
start = date(2005, 3, 10)
referenceDate = date(2005, 3, 10)

myScheduler = Scheduler()
ReferenceDateList = myScheduler.getSchedule(start=referenceDate,end=trim_end,freq="1M", referencedate=referenceDate)
# Create Simulator
xOIS = [ 3.0,  0.07536509, -0.208477,  0.07536509]
myVasicek = MC_Vasicek_Sim(datelist = [trim_start,trim_end],x = xOIS,simNumber = simNumber,t_step =1/365.0 )
#myVasicek.setVasicek(x=xOIS,minDay=trim_start,maxDay=trim_end,simNumber=simNumber,t_step=1/365.0)
myVasicek.getLibor()

# Create Coupon Bond with several startDates.
SixMonthDelay = myScheduler.extractDelay("6M")
TwoYearsDelay = myScheduler.extractDelay("2Y")
startDates = [referenceDate + nprnd.randint(0,3)*SixMonthDelay for r in range(10)]

# For debugging uncomment this to choose a single date for the forward bond
# print(startDates)
startDates = [date(2005,3,10)+SixMonthDelay,date(2005,3,10)+TwoYearsDelay ]
maturities = [(x+TwoYearsDelay) for x in startDates]

myPortfolio = {}
coupon = 0.07536509
for i in range(len(startDates)):
    notional=(-1.0)**i
    myPortfolio[i] = CouponBond(fee=1.0,start=startDates[i],coupon=coupon,notional=notional,
                                maturity= maturities[i], freq="3M", referencedate=referenceDate)

portfolioScheduleOfCF = set(ReferenceDateList)
for i in range(len(myPortfolio)):
    portfolioScheduleOfCF=portfolioScheduleOfCF.union(myPortfolio[i].getScheduleComplete()[0]
)
portfolioScheduleOfCF = sorted(portfolioScheduleOfCF.union(ReferenceDateList))
OIS = myVasicek.getSmallLibor(datelist=portfolioScheduleOfCF)
# at this point OIS contains all dates for which the discount curve should be known.
# If the OIS doesn't contain that date, it would not be able to discount the cashflows and the calcualtion would faill.

pvs={}
for t in portfolioScheduleOfCF:
    pvs[t] = np.zeros([1,simNumber])
    for i in range(len(myPortfolio)):
        myPortfolio[i].setLibor(OIS)
        pvs[t] = pvs[t] + myPortfolio[i].getExposure(referencedate=t).values

print(pvs)
pvsPlot = pd.DataFrame.from_dict(list(pvs.items()))
pvsPlot.index= list(pvs.keys())
pvs1={}
for i,t in zip(pvsPlot.values,pvsPlot.index):
    pvs1[t]=i[1][0]
pvs = pd.DataFrame.from_dict(data=pvs1,orient="index")
ax=pvs.plot(legend=False)
ax.set_xlabel("Year")
ax.set_ylabel("Coupon Bond Exposure")