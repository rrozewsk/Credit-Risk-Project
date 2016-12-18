from scipy.stats import norm, mvn #normal and bivariate normal
from numpy       import sqrt
from numpy import exp
from datetime import date, timedelta
import datetime as dt
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
        #self.maturity = end_date
        self.rating = rating

        self.CDSClass = CDS(start_date=start_date, end_date=end_date, freq=freq, coupon=1, referenceDate=referenceDate,
                      rating=rating, R=R)
        self.portfolioScheduleOfCF,self.datelist = self.CDSClass.getScheduleComplete()

    def getQ(self,referenceDate,end_date,rating,R,freq):
        ## Use CorporateDaily to get Q for referencedates ##
        # print("GET Q")
        # print(self.portfolioScheduleOfCF)
        myQ = CorporateRates()
        myQ.getCorporatesFred(trim_start=referenceDate, trim_end=end_date)
        ## Return the calculated Q(t,t_i) for bonds ranging over maturities for a given rating
        daterange = pd.date_range(start=referenceDate, end=end_date).date
        self.myQ = myQ.getCorporateQData(rating=rating, datelist=daterange, R=R)

        ## Only keep the 1Month Q #####
        #print(self.freq)
        self.Q1M = self.myQ[freq]
        #print(Q1M)

        ## Only for the dates specified
        #timed = self.portfolioScheduleOfCF[self.portfolioScheduleOfCF.index(self.referenceDate):]
        #self.Q1M = Q1M.loc[timed]
        self.Q1M = np.cumprod(self.Q1M)
        self.Q1M = self.Q1M/self.Q1M[self.referenceDate]
        #print(self.Q1M)

        return(self.Q1M)

    def Q_lhp(self,t, K1, K2, R, beta, Q):
        """Calculates the Tranche survival curve for the LHP model.

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
        if Q(t) == 1:
            return 1  # prevents infinity

        def emin(K):
            # Calculates E[min(L(T), K)] in LHP model
            C = norm.ppf(1 - Q(t))
            A = (1 / beta) * (C - sqrt(1 - beta * beta) * norm.ppf(K / (1 - R)))
            return (1 - R) * mvn.mvndst(upper=[C, -1 * A],
                                        lower=[0, 0],
                                        infin=[0, 0],  # set lower bounds = -infty
                                        correl=-1 * beta)[1] + K * norm.cdf(A)

        return 1 - (emin(K2) - emin(K1)) / (K2 - K1)

    def Q_gauss(self,t, K1, K2, Fs, Rs, betas, Qs):
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
        print('#', end="")
        Cs = [norm.ppf(1 - q(t)) for q in Qs]
        N = len(Fs)

        def f(K, z):
            ps = [norm.cdf((C - beta * z) / sqrt(1 - beta ** 2)) for C, beta in zip(Cs, betas)]
            mu = 1 / N * sum([p * F * (1 - R) for p, F, R in zip(ps, Fs, Rs)])
            sigma_squared = 1 / N / N * sum([p * (1 - p) * F ** 2 * (1 - R) ** 2
                                             for p, F, R in zip(ps, Fs, Rs)])
            sigma = sqrt(sigma_squared)
            return -1 * sigma * norm.pdf((mu - K) / sigma) - (mu - K) * norm.cdf((mu - K) / sigma)

        emin = lambda K: quad(lambda z: norm.pdf(z) * f(K, z), -10, 10)[0]
        return 1 - (emin(K2) - emin(K1)) / (K2 - K1)

    def Q_adjbinom(self,t, K1, K2, Fs, Rs, betas, Qs):
        """ Calculates the tranche survival probability under the adjusted
            binomial model.

            Arguments:
                t (datetime.date): Time.
                K1 (float): Starting tranche value (0 to 1).
                K2 (float): Final tranche value (0 to 1).
                Fs (list): List of fractional face values (floats) for each credit.
                Rs (list): List of recovery rates (floats) for each credit.
                betas (list): List of correlation perameters
                Qs (list): List of survival probabilities. These are callable functions that
                    takes in a single datetime.date argument and returns a float.
            Returns:
                float: The value of the tranche survival curve.
        """
        print(t)
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

    def gcd(self,a, b):
        if a * b == 0:
            return max(a, b)
        if a < b:
            return self.gcd(a, b % a)
        return self.gcd(a % b, b)

    def Q_exact(self,t, K1, K2, Fs, Rs, betas, Qs):
        Cs = [norm.ppf(1 - Q(t)) for Q in Qs]
        N = len(Fs)
        g = round(3600 * Fs[0] * (1 - Rs[0]))
        for j in range(1, N):
            g = self.gcd(g, round(3600 * Fs[j] * (1 - Rs[j])))
        g = g / 3600
        ns = [round(F * (1 - R) / g) for F, R in zip(Fs, Rs)]

        def f(j, k, z):
            if (j, k) == (0, 0):
                return 1.0
            if j == 0:
                return 0.0
            ps = [norm.cdf((C - beta * z) / sqrt(1 - beta ** 2)) for C, beta in zip(Cs, betas)]
            if k < ns[j - 1]:
                return f(j - 1, k, z) * (1 - ps[j - 1])
            return f(j - 1, k, z) * (1 - ps[j - 1]) + f(j - 1, k - ns[j - 1], z) * ps[j - 1]

        I = [quad(lambda z: norm.pdf(z) * f(N, k, z), -12, 12)[0] for k in range(sum(ns))]
        emin = lambda K: sum([I[k] * min(k * g, K) for k in range(sum(ns))])
        return 1 - (emin(K2) - emin(K1)) / (K2 - K1)


    def CDOPortfolio(self):
        ## Create a portfolio of CDSs ###
        N = 2 #Number of CDSs
        K1 = 0.03
        K2  = 0.07
        Fs = [0.5, 0.5]
        Rs = [0.40, 0.40]
        betas = [0.40, 0.40]
        # Should assume same start,reference and end_dates
        start_dates = [date(2012,2,28),date(2012,2,28)]
        referenceDates = [date(2013, 1, 20),date(2013, 1, 20)]
        end_dates = [date(2013, 12, 31),date(2013, 12, 31)]
        ratings = ["AAA","AAA"]
        freqs = ["3M","3M"]

        CDSClass = CDS(start_date=start_dates[0], end_date=end_dates[0], freq=freqs[0], coupon=1, referenceDate=referenceDates[0],
                            rating=ratings[0], R=Rs[0])

        ## price CDOs using LHP
        self.referenceDate = referenceDates[0]
        Q_now1 = self.getQ(referenceDates[0],end_dates[0],ratings[0],Rs[0],freqs[0])
        #Q_now2 = self.getQ(referenceDates[1], end_dates[1], ratings[1], Rs[1], freqs[1])
        print("Q_now")
        print(Q_now1)

        ### Create Q dataframe
        tvalues = Q_now1.index.tolist()
        Cs = pd.Series(Q_now1,index=tvalues)
        for cds_num in range(1,len(Fs)):
            Q_add = self.getQ(referenceDates[cds_num], end_dates[cds_num], ratings[cds_num], Rs[cds_num], freqs[cds_num])
            Q_add = pd.Series(Q_add,index = tvalues)
            Cs = pd.concat([Cs,Q_add],axis = 1)

        def expdecay(n):
            return lambda t: Cs.ix[t,n]

        ##
        #Qs = [expdecay(0),expdecay(1)]
        Qs = [expdecay(n) for n in range(0,Cs.shape[1])]

        ###### LHP Method #####################################################################

        lhpcurve = [self.Q_lhp(t, K1, K2, R = Rs[0], beta = betas[0], Q = Qs[0]) for t in tvalues]
        lhpcurve = pd.Series(lhpcurve,index = tvalues)
        #lhpcurve = np.cumprod(lhpcurve)
        #lhpcurve = lhpcurve/lhpcurve[tvalues[0]]

        CDSClass.setQ(lhpcurve)
        ProtectionLeg = CDSClass.getProtectionLeg()
        PremiumLeg = CDSClass.getPremiumLegZ()
        spreads = ProtectionLeg / PremiumLeg
        print("The spread for LHP is: ", 10000 * spreads, ".")
        ########################################################################################
        print('Gaussian progression: ', end="")
        gaussiancurve = [self.Q_gauss(t, K1, K2, Fs = Fs, Rs =Rs, betas=betas, Qs=Qs) for t in tvalues]
        gaussiancurve = pd.Series(gaussiancurve, index=tvalues)
        #gaussiancurve = np.cumprod(gaussiancurve)
        #gaussiancurve = gaussiancurve / gaussiancurve[tvalues[0]]

        CDSClass.setQ(gaussiancurve)
        ProtectionLeg = CDSClass.getProtectionLeg()
        PremiumLeg = CDSClass.getPremiumLegZ()
        spreads = ProtectionLeg / PremiumLeg
        print("The Gaussian spread is: ", 10000 * spreads, ".")

        adjustedbinomialcurve = [self.Q_adjbinom(t, K1, K2, Fs = Fs, Rs = Rs, betas=betas, Qs=Qs) for t in tvalues]
        adjustedbinomialcurve = pd.Series(adjustedbinomialcurve, index=tvalues)
        #adjustedbinomialcurve = np.cumprod(adjustedbinomialcurve)
        #adjustedbinomialcurve = adjustedbinomialcurve / adjustedbinomialcurve[tvalues[0]]

        CDSClass.setQ(adjustedbinomialcurve)
        ProtectionLeg = CDSClass.getProtectionLeg()
        PremiumLeg = CDSClass.getPremiumLegZ()
        spreads = ProtectionLeg / PremiumLeg
        print("The Adjusted Binomial spread is: ", 10000 * spreads, ".")

        exactcurve = [self.Q_exact(t, K1, K2, Fs =Fs, Rs = Rs, betas =betas, Qs =Qs) for t in tvalues]
        exactcurve = pd.Series(exactcurve, index=tvalues)
        #exactcurve = np.cumprod(exactcurve)
        #exactcurve = exactcurve / exactcurve[tvalues[0]]

        CDSClass.setQ(exactcurve)
        ProtectionLeg = CDSClass.getProtectionLeg()
        PremiumLeg = CDSClass.getPremiumLegZ()
        spreads = ProtectionLeg / PremiumLeg
        print("The Exact spread is: ", 10000 * spreads, ".")






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
test.CDOPortfolio()
#print(test.getQ())
#test.Q_lhp(K1=K1, K2 = K2, R =0.4,beta = 0.30)

### Can be extended to a portfolio of CDOs ###

