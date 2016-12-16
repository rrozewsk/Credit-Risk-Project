from scipy.stats import norm, mvn #normal and bivariate normal
from numpy       import sqrt
from numpy import exp
from datetime import date, timedelta
import pandas as pd
import numpy as np

from Curves.Corporates.CorporateDaily import CorporateRates
from Products.Credit.CDS import CDS
from Scheduler.Scheduler import Scheduler
from scipy.integrate import quad
from Curves.PortfolioLoss.ExactFunc import ExactFunc


class PortfolioLossCalculation(object):
    def __init__(self, K1, K2, Fs, Rs, betas,start_date,end_date,freq,coupon,referenceDate,rating,R=.4):
        self.K1 = K1
        self.K2 = K2
        self.Fs= Fs
        self.Rs = Rs
        self.betas = betas
        #self.Qs = Qs
        self.referenceDate = referenceDate
        self.freq = freq
        self.coupon = coupon
        self.R = R
        self.start_date = start_date
        self.end_date = end_date
        self.myScheduler = Scheduler()
        delay = self.myScheduler.extractDelay(self.freq)
        self.maturity = end_date
        self.rating = rating

        self.CDSClass = CDS(start_date=start_date, end_date=end_date, freq=freq, coupon=1, referenceDate=referenceDate,
                      rating=rating, R=R)
        self.portfolioScheduleOfCF,self.datelist = self.CDSClass.getScheduleComplete()

    def getQ(self):
        ## Use CorporateDaily to get Q for referencedates ##
        # print("GET Q")
        # print(self.portfolioScheduleOfCF)
        myQ = CorporateRates()
        myQ.getCorporatesFred(trim_start=self.referenceDate, trim_end=self.end_date)
        ## Return the calculated Q(t,t_i) for bonds ranging over maturities for a given rating
        daterange = pd.date_range(start=self.referenceDate, end=self.maturity).date
        self.myQ = myQ.getCorporateQData(rating=self.rating, datelist=daterange, R=self.R)

        ## Only keep the 1Month Q #####
        #print(self.freq)
        Q1M = self.myQ[self.freq]
        #print(Q1M)

        ## Only for the dates specified
        timed = self.portfolioScheduleOfCF[self.portfolioScheduleOfCF.index(self.referenceDate):]
        self.Q1M = Q1M.loc[timed]
        self.Q1M = np.cumprod(self.Q1M)
        #self.Q1M = self.Q1M/self.Q1M[self.referenceDate]
        #print(self.Q1M)

        return(self.Q1M)

    def emin(self,K,Q,beta,R):
        # Calculates E[min(L(T), K)] in LHP model
        C = norm.ppf(1 - Q)
        A = (1 / beta) * (C - sqrt(1 - beta * beta) * norm.ppf(K / (1 - R)))

        emin =  (1 - R) * mvn.mvndst(upper=[C, -1 * A],
                                    lower=[0, 0],
                                    infin=[0, 0],  # set lower bounds = -infty
                                    correl=-beta)[1] + K * norm.cdf(A)
        return(emin)

    def Q_lhp(self,K1, K2, R, beta):
        """
        Calculates the Tranche survival curve for the LHP model.

        Args:
            T (datetime.date): Should be given in "days" (no hours, minutes, etc.)
            K1 (float): The starting value of the tranche. Its value should be
                between 0 & 1.
            K2 (float): The final value of the tranche.
            beta (float): Correlation perameter. Its value should be between 0 and 1.
            R (float): Recovery rate. Usually between 0 and 1.
            Q (callable function): A function Q that takes in a dateime.date and
                outputs a float from 0 to 1. It is assumed that Q(0) = 1 and Q is
                decreasing. Represents survival curve of each credit.
        """
        print("Q_LHP")

        ## Loop through dates in  Q ###
        # Q is a dataframe with dates and values
        Q_now = self.getQ()
        print(Q_now)
        QTranche = []

        count = 1
        for row in Q_now.iteritems():
            Q_temp = row[1]
            #print(Q_temp)
            emin1 = self.emin(K = K1,Q = Q_temp,beta = beta, R = R)
            emin2 = self.emin(K = K2,Q = Q_temp,beta = beta, R = R)
            QTranche.append(1 - (emin2 - emin1) / (K2 - K1))
            count += 1

        QTranche = pd.Series(QTranche,index = Q_now.index,name=self.freq)
        #QTranche=pd.DataFrame(QTranche.values,index=Q_now.index,columns=list(self.freq))
        QTranche = QTranche/QTranche[self.referenceDate]
        ## Data ready to be used in pricing ###
        Q_New = pd.concat([Q_now, QTranche], axis=1)
        Q_New.columns = ["3MQ","3M"]
        #Q_now[self.freq] = QTranche
        print(Q_New)
        ## Calculate Spread ###
        self.CDSClass.setQ(Q_New)
        ProtectionLeg = self.CDSClass.getProtectionLeg()
        PremiumLeg = self.CDSClass.getPremiumLegZ()

        spreads = ProtectionLeg/PremiumLeg

        print("SPREADS BP")
        print("Spread: ",spreads*10000,"Bp")

        return Q_New

    def Q_gauss(self, K1, K2, Fs, Rs, betas, Qs):
        """ Calculate the tranche survival probability in the Gaussian heterogenous model.

        Arguments:
            t (float): Time. Positive.
            K1 (float): Starting tranche value. Between 0 and 1.
            K2 (float): Ending tranche value. Between 0 and 1.
            Fs (list): List of fractional face values for each credit. Each entry must be
                between 0 and 1.
            Rs (list): List of recovery rates for each credit. Each entry must be between
                0 and 1.
            betas (list): Correlation perameters for each credit. Each entry must be between
                0 and 1.
            Qs (list): Survival curves for each credit. Each entry must be a callable function
                that takes in a datetime.date argument and returns a number from 0 to 1.
        """
        ## Does not work, need to get Qs for several credits #####
        Q_now = self.getQ()

        Cs = [norm.ppf(1 - q(t)) for q in Qs]
        N = len(Fs)

        def f(z):
            ps = [norm.cdf((C - beta * z) / sqrt(1 - beta * beta)) for C, beta in zip(Cs, betas)]
            mu = 1 / N * sum([p * F * (1 - R) for p, F, R in zip(ps, Fs, Rs)])
            sigma_squared = 1 / N / N * sum([p * (1 - p) * F ** 2 * (1 - R) ** 2 for p, F, R in zip(ps, Fs, Rs)])
            sigma = sqrt(sigma_squared)
            return (sigma * norm.pdf((mu - K1) / sigma)
                    - sigma * norm.pdf((mu - K2) / sigma)
                    + (mu - K1) * norm.cdf((mu - K1) / sigma)
                    - (mu - K2) * norm.cdf((mu - K2) / sigma))

        integral = quad(lambda z: f(z) * norm.pdf(z), -10, 10)[0]
        return 1 - integral / (K2 - K1)

    def Q_adjbinom(self, K1, K2, Fs, Rs, betas, Qs):
        if Qs[0](t) == 1:
            return 1.0  # initial value -- avoids weird nan return
        N = len(Fs)
        Cs = [norm.ppf(1 - Q(t)) for Q in Qs]
        L = sum([(1 - R) * F for R, F in zip(Rs, Fs)]) / N

        def choose(n, k):  # Calculates binomial coeffecient: n choose k.
            if k == 0 or k == n:
                return 1
            return choose(n - 1, k - 1) + choose(n - 1, k)

        def g(k, z):
            ps = [norm.cdf((C - beta * z) / sqrt(1 - beta * beta)) for C, beta in zip(Cs, betas)]
            p_avg = sum([(1 - R) * F / L * p for R, F, p in zip(Rs, Fs, ps)]) / N
            f = lambda k: choose(N, k) * p_avg ** k * (1 - p_avg) ** (N - k)
            vA = p_avg * (1 - p_avg) / N
            vE = 1 / N / N * sum([((1 - R) * F / L) ** 2 * p * (1 - p) for R, F, p in zip(Rs, Fs, ps)])
            m = p_avg * N
            l = int(m)
            u = l + 1
            o = (u - m) ** 2 + ((l - m) ** 2 - (u - m) ** 2) * (u - m)
            alpha = (vE * N + o) / (vA * N + o)
            if k == l:
                return f(l) + (1 - alpha) * (u - m)
            if k == u:
                return f(u) - (1 - alpha) * (l - m)
            return alpha * f(k)

        I = lambda k: quad(lambda z: norm.pdf(z) * g(k, z), -10, 10)[0]
        emin = lambda K: sum([I(k) * min(L * k, K) for k in range(0, N + 1)])
        return 1 - (emin(K2) - emin(K1)) / (K2 - K1)


    def ExactFunc(self):
        ### Exact function from Ryan ####
        HowExact = ExactFunc()


#def expdecay(today, rate):
#    return lambda t: exp(-1 * (t - today).days / 365 * rate)
#today = date(2012,1, 1)
#Q = expdecay(today, 0.0140)
#print(Q(date(2012,1, 3)))
#print(Q(date(2012,5, 3)))

t_step = 1.0 / 365.0
simNumber = 10
start_date = date(2012,2,28)
end_date = date(2015,12,31)  # Last Date of the Portfolio
referenceDate = date(2012, 12, 20)

#tvalues = [today + timedelta(days = 30) * n for n in range(37)] #3 years
#print(tvalues)

K1 = 0.01
K2 = 0.03
Fs = [0.3, 0.8]
Rs = [0.40, 0.60]
betas = [0.30, 1]
freq = "3M"
test = PortfolioLossCalculation(K1 = K1, K2 = K2, Fs = Fs, Rs =Rs, betas = betas,
                                start_date = start_date,end_date = end_date,freq=freq,
                                coupon = 0.001,referenceDate = referenceDate,rating="AAA",
                                R=0)
test.getQ()
test.Q_lhp(K1=K1, K2 = K2, R =0.4,beta = 0.30)

### Can be extended to a portfolio of CDOs ###

