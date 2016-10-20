from unittest import TestCase
from MonteCarloSimulators.Vasicek.vasicekMCSim import MC_Vasicek_Sim
from Products.Rates.CouponBond import CouponBond
from Scheduler.Scheduler import Scheduler
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

# dr(t) = k(θ − r(t))dt + σdW(t)
# self.kappa = x[0]
# self.theta = x[1]
# self.sigma = x[2]
# self.r0 = x[3]
myScheduler = Scheduler()

observationdate = minDay = date(2005,1,10) # Observation Date
maxDay = date(2010,1,10)  # Last Date of the Portfolio
start = date(2005, 3, 30)
maturity = start+ myScheduler.extractDelay("1Y")
referenceDate = date(2005, 5, 3)  # 6 months after trim_start
simNumber = 5
R = 0.4
inArrears = True
freq = '3M'
t_step = 1.0/365

simNumber=5
coupon = 0.08
fee= 1.0
xR = [3.0, 0.05, 0.04, 0.03]

myBond = CouponBond(fee=fee, start=start, maturity=maturity, coupon=coupon, freq="3M", referencedate=referenceDate, observationdate = observationdate)
fulllist, datelist = myBond.getScheduleComplete()
myMC = MC_Vasicek_Sim(datelist=[minDay,maxDay],x=xR,simNumber=simNumber,t_step=t_step)
#myMC.setVasicek(x=xR, minDay=minDay, maxDay=maxDay, simNumber=simNumber, t_step=t_step)
myMC.getLibor()
libor = myMC.getSmallLibor(tenors=fulllist)
myBond.setLibor(libor)



class TestBond(TestCase):
    def test01_PV(self):
        myPV = myBond.getPV(referencedate=referenceDate)
        print(myPV)
        print(myBond.pv)

    def test00_getLiborAvg(self):
        print(myBond.getLiborAvg(yieldIn=0.05,datelist=datelist))

    def test02_getYield(self):
        x=[]
        y=[]
        for i in range(95,101):
            xx=0.01*i
            y.append(i)
            price = xx
            yieldOut = myBond.getYield(price)
            x.append(yieldOut)

        fig, ax = plt.subplots()
        x=pd.DataFrame(data=x, index=y)
        print(x)
        for xx in x.columns:
            ax.plot(x[xx]+xx*0.001,y)
        ax.grid(True)
        ticklines = ax.get_xticklines() + ax.get_yticklines()
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

        for line in ticklines:
            line.set_linewidth(3)

        for line in gridlines:
            line.set_linestyle('-')

        for label in ticklabels:
            label.set_color('r')
            label.set_fontsize('medium')

        plt.xlabel("Yield")
        plt.ylabel("CouponBond Price")
        plt.show()
        a=1
