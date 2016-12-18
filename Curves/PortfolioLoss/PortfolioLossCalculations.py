from scipy.stats import norm, mvn #normal and bivariate normal
from numpy       import sqrt
from numpy import exp
from datetime import date, timedelta
import datetime as dt
import pandas as pd
import numpy as np

from parameters import x0Vas
from Curves.Corporates.CorporateDaily import CorporateRates
from Products.Credit.CDS import CDS
from Scheduler.Scheduler import Scheduler
from scipy.integrate import quad
from Curves.PortfolioLoss.ExactFunc import ExactFunc
from parameters import xR,t_step, simNumber

from Boostrappers.CDSBootstrapper.CDSVasicekBootstrapper import BootstrapperCDSLadder
from MonteCarloSimulators.Vasicek.vasicekMCSim import MC_Vasicek_Sim

class PortfolioLossCalculation(object):
    def __init__(self, K1, K2, Fs, Rs, betas,start_dates,end_dates,freqs,coupons,referenceDates,ratings,bootstrap):
        self.bootstrap = bootstrap
        self.K1 = K1
        self.K2 = K2
        self.Fs= Fs
        self.Rs = Rs
        self.betas = betas
        self.referenceDates = referenceDates
        self.freqs = freqs
        self.coupons = coupons
        self.start_dates = start_dates
        self.end_dates = end_dates
        self.ratings = ratings

        self.R = Rs[0]
        self.beta = betas[0]
        self.referenceDate = referenceDates[0]
        self.freq = freqs[0]
        self.coupon = coupons[0]
        self.start_date = start_dates[0]
        self.end_date = end_dates[0]
        self.rating = ratings[0]
        self.maturity = end_dates[0]

        self.myScheduler = Scheduler()

    def setParameters(self,R,rating,coupon,beta,start_date,referenceDate,end_date,freq):
        self.R = R
        self.beta = beta
        self.referenceDate = referenceDate
        self.freq = freq
        self.coupon = coupon
        self.start_date = start_date
        self.end_date = end_date
        self.rating = rating
        self.maturity = end_date

    def getQ(self,start_date,referenceDate,end_date,freq,coupon,rating,R):
        ## Use CorporateDaily to get Q for referencedates ##
        # print("GET Q")
        # print(self.portfolioScheduleOfCF)

        if self.bootstrap:
            print("Q bootstrap")
            CDSClass = CDS(start_date=start_date, end_date=end_date, freq=freq, coupon=coupon,
                           referenceDate=referenceDate,rating=rating, R=R)
            myLad = BootstrapperCDSLadder(start=self.start_date, freq=[freq], CDSList=[CDSClass],
                                          R=CDSClass.R).getXList(x0Vas)[freq]
            self.Q1M = MC_Vasicek_Sim(x=myLad, t_step = 1 / 365,
                                 datelist=[CDSClass.referenceDate, CDSClass.end_date],simNumber=simNumber).getLibor()[0]
            print(self.Q1M)
        else:
            myQ = CorporateRates()
            myQ.getCorporatesFred(trim_start=referenceDate, trim_end=end_date)
            ## Return the calculated Q(t,t_i) for bonds ranging over maturities for a given rating
            daterange = pd.date_range(start=referenceDate, end=end_date).date
            myQ = myQ.getCorporateQData(rating=rating, datelist=daterange, R=R)
            Q1M = myQ[freq]
            print(Q1M)


        return(Q1M)

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

    def getZ_Vasicek(self):
        ### Get Z(t,t_i) for t_i in datelist #####

        ## Simulate Vasicek model with paramters given in workspace.parameters
        # xR = [5.0, 0.05, 0.01, 0.05]
        # kappa = x[0]
        # theta = x[1]
        # sigma = x[2]
        # r0 = x[3]
        vasicekMC = MC_Vasicek_Sim(datelist=[self.referenceDate, self.end_date], x=xR, simNumber=10, t_step=t_step)
        self.myZ = vasicekMC.getLibor()
        self.myZ = self.myZ.loc[:, 0]

    def getScheduleComplete(self):
        datelist = self.myScheduler.getSchedule(start=self.start_date, end=self.maturity, freq=self.freq,
                                                     referencedate=self.referenceDate)
        ntimes = len(datelist)
        fullset = list(sorted(list(set(datelist)
                                   .union([self.referenceDate])
                                   .union([self.start_date])
                                   .union([self.maturity])
                                   .union([self.referenceDate])
                                   )))
        return fullset, datelist

    def getPremiumLegZ(self,myQ):
        Q1M = myQ
        # Q1M = self.myQ["QTranche"]
        fulllist, datelist = self.getScheduleComplete()
        portfolioScheduleOfCF = fulllist
        timed = portfolioScheduleOfCF[portfolioScheduleOfCF.index(self.referenceDate):]

        Q1M = Q1M.loc[timed]

        zbarPremLeg = self.myZ / self.myZ.loc[self.referenceDate]
        zbarPremLeg = zbarPremLeg.loc[timed]
        ## Calculate Q(t_i) + Q(t_(i-1))
        Qplus = []
        out = 0
        for i in range(1, len(Q1M)):
            out = out + (Q1M[(i - 1)] + Q1M[i]) * float((timed[i] - timed[i - 1]).days / 365) * zbarPremLeg[i]

        zbarPremLeg = pd.DataFrame(zbarPremLeg, index=timed)
        # print("Premium LEg")
        PVpremiumLeg = out * (1 / 2)
        # print(PVpremiumLeg)
        ## Get coupon bond ###
        print(PVpremiumLeg)
        return PVpremiumLeg

        # //////////////// Get Protection leg Z(t_i)( Q(t_(i-1)) - Q(t_i) )

    def getProtectionLeg(self,myQ):
        print("Protection Leg ")
        Q1M = myQ
        #Qminus = np.gradient(Q1M)
        zbarProtectionLeg = self.myZ

        out = 0
        for i in range(1,zbarProtectionLeg.shape[0]):
            #out = -Qminus[i] * zbarProtectionLeg.iloc[i]*float(1/365)
            out = out  -(Q1M.iloc[i]-Q1M.iloc[i-1]) * zbarProtectionLeg.iloc[i]

        ## Calculate the PV of the premium leg using the bond class
        print(out)
        return out

    def CDOPortfolio(self):
        self.setParameters(R=self.Rs[0], rating = self.ratings[0], coupon=self.coupons[0],
                           beta = self.betas[0], start_date = start_dates[0],
                           referenceDate = referenceDates[0], end_date = end_dates[0], freq = self.freqs[0])
        ## price CDOs using LHP
        Q_now1 = self.getQ(start_date = self.start_dates[0],referenceDate=self.referenceDates[0],
                           end_date = self.end_dates[0],freq = self.freqs[0],coupon=self.coupons[0],
                           rating = self.ratings[0],R = self.Rs[0])

        ## Estimate default probabilites from Qtranche list ###
        ## Assume that lambda is constant over a small period of time
        def getApproxDefaultProbs(Qvals, freq, tvalues):
            t_start = tvalues[0]
            delay = self.myScheduler.extractDelay(freq)
            delay_days = ((t_start + delay) - t_start).days
            ## Estimate constant lambda
            lam = -(1 / delay_days) * np.log(Qvals[t_start])
            Qvals = [((Qvals[t_start] * exp(-lam * (t - t_start).days)) / Qvals[t]) for t in tvalues]
            return (Qvals)
        ########################################################


        print(Q_now1)
        ### Create Q dataframe
        tvalues = Q_now1.index.tolist()
        Cs = pd.Series(Q_now1,index=tvalues)
        for cds_num in range(1,len(Fs)):
            Q_add = self.getQ(start_date = self.start_dates[cds_num],referenceDate=self.referenceDates[cds_num],
                           end_date = self.end_dates[cds_num],freq = self.freqs[cds_num],coupon=self.coupons[cds_num],
                           rating = self.ratings[cds_num],R = self.Rs[cds_num])
            Q_add = pd.Series(Q_add,index = tvalues)
            Cs = pd.concat([Cs,Q_add],axis = 1)

        def expdecay(n):
            return lambda t: Cs.ix[t,n]

        ##
        #Qs = [expdecay(0),expdecay(1)]
        Qs = [expdecay(n) for n in range(0,Cs.shape[1])]

        self.getZ_Vasicek()

        ###### LHP Method #####################################################################
        Rs_mean = np.mean(self.Rs)
        betas_mean = np.mean(betas)

        lhpcurve = [self.Q_lhp(t, self.K1, self.K2, R = Rs[0], beta = betas[0], Q = Qs[0]) for t in tvalues]

        lhpcurve = pd.Series(lhpcurve, index=tvalues)
        lhpcurve = getApproxDefaultProbs(lhpcurve,freq=self.freq,tvalues=tvalues)
        lhpcurve = pd.Series(lhpcurve, index=tvalues)

        ProtectionLeg = self.getProtectionLeg(myQ = lhpcurve)
        PremiumLeg = self.getPremiumLegZ(myQ = lhpcurve)
        spreads = ProtectionLeg / PremiumLeg
        print("The spread for LHP is: ", 10000 * spreads, ".")
        ########################################################################################

        ###### Gaussian Method #####################################################################
        print('Gaussian progression: ', end="")
        gaussiancurve = [self.Q_gauss(t, self.K1, self.K2, Fs = self.Fs, Rs =self.Rs, betas=self.betas, Qs=Qs) for t in tvalues]
        gaussiancurve = pd.Series(gaussiancurve, index=tvalues)
        gaussiancurve = getApproxDefaultProbs(gaussiancurve, freq=self.freq, tvalues=tvalues)
        gaussiancurve = pd.Series(gaussiancurve, index=tvalues)

        ProtectionLeg = self.getProtectionLeg(myQ=gaussiancurve)
        PremiumLeg = self.getPremiumLegZ(myQ=gaussiancurve)
        spreads = ProtectionLeg / PremiumLeg
        print("The spread for Gaussian is: ", 10000 * spreads, ".")
        ########################################################################################

        ###### Adjusted Binomial Method #####################################################################
        adjustedbinomialcurve = [self.Q_adjbinom(t, self.K1, self.K2, Fs = self.Fs, Rs = self.Rs, betas=self.betas, Qs=Qs) for t in tvalues]
        adjustedbinomialcurve = pd.Series(adjustedbinomialcurve, index=tvalues)
        adjustedbinomialcurve = getApproxDefaultProbs(adjustedbinomialcurve, freq=self.freq, tvalues=tvalues)
        adjustedbinomialcurve = pd.Series(adjustedbinomialcurve, index=tvalues)
        #adjustedbinomialcurve = adjustedbinomialcurve.to_frame(self.freqs[0])

        ProtectionLeg = self.getProtectionLeg(myQ=adjustedbinomialcurve)
        PremiumLeg = self.getPremiumLegZ(myQ=adjustedbinomialcurve)
        spreads = ProtectionLeg / PremiumLeg
        print("The spread for Ajusted Binomial is: ", 10000 * spreads, ".")
        ########################################################################################

        ###### Exact Method #####################################################################
        exactcurve = [self.Q_exact(t, self.K1, self.K2, Fs =self.Fs, Rs = self.Rs, betas =self.betas, Qs =Qs) for t in tvalues]
        exactcurve = pd.Series(exactcurve, index=tvalues)
        exactcurve = getApproxDefaultProbs(exactcurve, freq=self.freq, tvalues=tvalues)
        exactcurve = pd.Series(exactcurve, index=tvalues)
        #exactcurve = exactcurve.to_frame(self.freqs[0])

        ProtectionLeg = self.getProtectionLeg(myQ=exactcurve)
        PremiumLeg = self.getPremiumLegZ(myQ=exactcurve)
        spreads = ProtectionLeg / PremiumLeg
        print("The spread for Exact is: ", 10000 * spreads, ".")
        ########################################################################################


K1 = 0.00001
K2 = 0.03
Fs = [0.3, 0.5,0.2]
Rs = [0.40, 0.40,0.40]
betas = [0.30, 0.30,0.30]
bootstrap = True
# Should assume same start,reference and end_dates
start_dates = [date(2012, 2, 28), date(2012, 2, 28),date(2012, 2, 28)]
referenceDates = [date(2013, 1, 20), date(2013, 1, 20), date(2013, 1, 20)]
end_dates = [date(2013, 12, 31), date(2013, 12, 31),date(2013, 12, 31)]
ratings = ["AAA", "AAA","AAA"]
freqs = ["3M", "3M","3M"]
coupons = [1,1,1]

test = PortfolioLossCalculation(K1 = K1, K2 = K2, Fs = Fs, Rs =Rs, betas = betas,
                                start_dates = start_dates,end_dates = end_dates,freqs=freqs,
                                coupons = coupons,referenceDates = referenceDates,ratings=ratings,bootstrap = False)
test.CDOPortfolio()


