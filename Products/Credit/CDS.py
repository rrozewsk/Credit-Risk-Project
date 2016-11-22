__author__ = 'ryan rozewski'

import Scheduler.Scheduler as schedule
from MonteCarloSimulators.Vasicek.vasicekMCSim import MC_Vasicek_Sim
from Curves.Corporates.CorporateDaily import CorporateRates
from Products.Rates.CouponBond import CouponBond
from datetime import date
from parameters import xR, t_step,simNumber,Coupon

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
    def __init__(self,fee,start,end,freq,coupon,refDate,observationdate,rating,R):
        self.startDate=start
        self.endDate=end
        self.freq=freq
        self.R = R
        self.fee =fee
        self.rating = rating
        self.coupon=coupon
        self.refDate=refDate
        self.obsDate = observationdate
        ## Get datelist ##
        #dtlist = schedule.Scheduler()
        #self.datelist = dtlist.getDatelist(start=self.startDate, end=self.endDate, freq=self.freq,ref_date=self.refDate)
        self.myBond = CouponBond(fee=self.fee, coupon=self.coupon, start=self.startDate, maturity=self.endDate,
                            freq=self.freq, referencedate=self.refDate, observationdate=self.obsDate)
        self.fullset, self.datelist = self.myBond.getScheduleComplete()

    def getZ_Vasicek(self):
        ### Get Z(t,t_i) for t_i in datelist #####

        ## Simulate Vasicek model with paramters given in workspace.parameters
        vasicekMC = MC_Vasicek_Sim(datelist=self.datelist, x=xR, simNumber=simNumber, t_step=t_step)
        vasicekSim = vasicekMC.getLibor()
        #print(vasicekSim)
        ## get Libor Z for reference dates ##
        vasicekMCRef = vasicekMC.getSmallLibor(tenors = self.datelist)

        ## Only keep the MC simulation ##
        vasicekMCRef = vasicekMCRef.iloc[:,1]
        return(vasicekMCRef)

    def getQ_Corporate(self):
        ## Use CorporateDaily to get Q for referencedates ##
        myQ = CorporateRates()
        ## Return the calculated Q(t,t_i) for bonds ranging over maturities for a given rating
        corporateQ = myQ.getCorporateQData(rating=self.rating,datelist=self.datelist,R=self.R)
        return corporateQ

    def getZbar(self):
        myZ = self.getZ_Vasicek()
        myQ = self.getQ_Corporate()

        ## Calculate Zbar(t,t_i) for various maturities
        zbar = myQ
        for i in range(myQ.shape[1]):
            zbar.iloc[:, i] = myQ.iloc[:, i] * myZ
        return zbar

    def getPV(self,observationdate):
        ### Get the present value of the CDS at an observationDate ####
        ## Get Zbar ##
        zbar = self.getZbar()
        ### use the class CouponBond to Calculate the present value
        self.myBond.setLibor(zbar)
        pv = self.myBond.getPV(referencedate=self.refDate)
        print(pv)


#### TEST FUNCTIONS ###
#test = CDS(fee = 1,start =date(2005, 1, 20),end = date(2006,4,22), freq = '1M',coupon=Coupon,refDate=date(2005, 6, 30),
#           observationdate=date(2005,3,31) ,rating='AAA',R=0.9999)
#pv = test.getPV(date(2005,1,25))


## About referencedate and observationdate ##
## Referencedate is the a date close to the coupon date. If referencedate is chosen prior
## to a start_date + freq*i then start_date + freq*i is chosen as the start date in datelist
## ObservationDate is the date when the contract is observed


#dt = scheduler.Scheduler()
#dt_ex = dt.getDatelist(start = date(2015,2,2), end= date(2016,2,2),freq='1D',ref_date= date(2015,7,7))
## Libor calculations
#vasicek_libor = vasicek.MC_Vasicek_Sim(datelist=dt_ex,x=parameters.xR,simNumber=2,t_step=1)
#corporate_Q = CorporateDaily.CorporateRates()
#get_Q = corporate_Q.getCorporateQData(rating='AAA',datelist=dt_ex,R=0.4)

#print(vasicek_libor.getLibor())
#print(get_Q)
#vasicek.MC_Vasicek_Sim()


