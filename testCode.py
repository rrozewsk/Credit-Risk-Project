



#######
from datetime import timedelta

from scipy.stats import norm, mvn  # normal and bivariate normal
from numpy import sqrt



class CDS(object):
    """A generic CDS class. Can also be used for CDO tranches.

    Attributes:
        today (datetime.date): The present date.
        maturity (datetime.date): The maturity of the CDO.
        remaining_payments (list): A list of datetime.date objects indicating
            the remaining payment dates for the premium leg.
        R (float): The recovery fraction (0 to 1).
        Z (callable function): A function Z(t) that outputs the discount factor
            Z(t) for a given time t (datetime.date object). Input and output are
            both positive floats.
        Q (callable function): A function Q(t) that takes in a single
            datetime.date perameter and returns a float representing the tranche
            survival probability. This function should be well-defined for all
            dates between `today` and `maturity`.
    """

    def __init__(self, today, maturity, remaining_payments, R, Z, Q):
        self.today = today
        self.maturity = maturity
        self.remaining_payments = remaining_payments
        self.R = R
        self.Z = Z
        self.Q = Q

    def rpv01(self):
        """Returns the value of the tranche premium leg for unit notional.
        """
        days = [self.today] + self.remaining_payments
        print(days)
        nodes = [(day - self.today).days / 365 for day in days]
        # qvals = [self.Q(day) for day in days]
        qvals = self.Q
        total = 0
        for i in range(1, len(days)):
            delta = nodes[i] - nodes[i - 1]
            total += delta * self.Z(days[i]) * (qvals[i] + qvals[i - 1])
        return total / 2

    def protectionLegPV(self, N=200):
        """Returns the value of the protection leg for unit notional.

        Arguments:
            N (int, optional): The number of nodes for calculating the integral.
        """
        delta = (self.maturity - self.today).days / N
        days = [today + timedelta(days=delta) * n for n in range(N + 1)]
        # print(days)
        # qvals = [self.Q(day) for day in days]
        qvals = self.Q
        values = [Z(days[i]) * (qvals[i - 1] - qvals[i])
                  for i in range(1, len(days))]
        return (1 - self.R) * sum(values)

    def parSpread(self, N=200):
        """ Returns the par spread.

        Arguments:
            N (int, optional): The number of nodes for calculating the
                protection leg integral.
        """
        return self.protectionLegPV(N) / self.rpv01()



def Q_lhp(t, K1, K2, R, beta, Q):
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


###### TEST functions #######
from numpy import exp
from datetime import date, timedelta
K1 = 0.03
K2 = 0.07
Fs = [0.3, 0.8]
Rs = [0.40, 0.60]
def expdecay(today, rate):
    return lambda t: exp(-1 * (t - today).days / 365 * rate)4
today = date(2012,1, 1)
Qs = [expdecay(today, 0.0120), expdecay(today, 0.0160)]
betas = [0.30, 0.40]
tvalues = [today + timedelta(days = 30) * n for n in range(37)] #3 years
# LHP perameters average the other perameters
R = 0.50
Q = expdecay(today, 0.0140)
beta = 0.35

# Takes FOREVER
lhpcurve = [Q_lhp(t, K1, K2, R, beta, Q) for t in tvalues]