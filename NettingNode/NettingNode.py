__author__ = 'marcopereira'
import numpy as np
import pandas as pd

from library.AffineModels.Exponential import Exponential
from library.AffineModels.VasicekModel import VasicekModel
from library.Boostrappers.CDSBootstrapper.CDSVasicekBootstrapper import  BootstrapperCDSLadder
from library.Curves.CurveDetails.CurveDetails import Bunch, dataType
from library.Instrument.Instrument import Instrument


###########################################################
###########################################################
###########################################################
###########################################################


class NettingNode(Instrument):
    counter = 0
    def __init__(self, Counterparty, QBankFunc, LiborFunc,
                            OISFunc, QFunc, model, myCorp, BootstrapperCDSLadder, start,
                 trim_start, trim_end, t_step, simNumber, R, periodsCDS, Country='US'):
        Instrument.__init__(self, start=start, Country=Country)
                # LiborFunc
        if (LiborFunc.type is not dataType.none):
            if (LiborFunc.type is dataType.function):
                if (hasattr(LiborFunc.function, '__call__')):
                    self.LiborFuncIsTrue = True
                    self.xL = LiborFunc.x

        # QFunc
        if (QFunc.type is not dataType.none):
            if (QFunc.type is dataType.function):
                if (hasattr(QFunc.function, '__call__')):
                    self.QFuncIsTrue = True
                    self.xQ = QFunc.x

        # OISFunc
        if (OISFunc.type is not dataType.none):
            if (OISFunc.type is dataType.function):
                if (hasattr(OISFunc.function, '__call__')):
                    self.OISFuncIsTrue = True
                    self.xR = OISFunc.x

        # QBankFunc
        if (QBankFunc.type is not dataType.none):
            if (QBankFunc.type is dataType.function):
                if (hasattr(QBankFunc.function, '__call__')):
                    self.QBankFuncIsTrue = True
                    self.xRBank = QBankFunc.x



        self.NettingNodeId = NettingNode.counter
        NettingNode.counter = NettingNode.counter+1
        self.trades = []
        self.model = model
        self.CDSBootstrapper = BootstrapperCDSLadder
        self.OIS = []
        self.FundingCurve = []
        self.myCorp = myCorp
        self.Counterparty = Counterparty
        self.EPE = {}
        self.ENE = {}
        self.CVA = {}
        self.DVA = {}
        self.FVA = {}
        self.OIS = []
        self.Q = []
        self.QBank = []
        self.BenefitFVA = {}
        self.R = R
        self.scheduleCF = []
        self.simNumber = simNumber
        self.start = start
        self.trim_start = trim_start
        self.trim_end = trim_end
        self.periodsCDS = periodsCDS
        self.t_step = t_step
        self.QBankFunc=QBankFunc
        self.LiborFunc=LiborFunc
        self.OISFunc=OISFunc
        self.QFunc=QFunc
        # CashFlow Dates
        self.myCorp = myCorp


    def getxQ(self, periodsCDS, quotesIn, Coupon=0.05):
        ###########################################################
        ###########################################################
        ###########################################################
        ###########################################################

        R = 0.4
        # Initial
        xR = [0.5, 0.0]
        simNumber = 1
        # Leg Preparation
        Exponential.trim_start = self.trim_start
        Exponential.simNumber = 1
        VasicekModel.trim_start = self.trim_start
        VasicekModel.simNumber = simNumber

        LiborDetails = Bunch()
        LiborDetails.type = dataType.none

        QDetails = Bunch()
        QDetails.type = dataType.function
        QDetails.function = VasicekModel(x=self.xQ).getLiborAvg
        QDetails.x = self.xQ
        OISDetails = Bunch()
        OISDetails.x = xR
        OISDetails.type = dataType.function
        OISDetails.function = Exponential(x=self.xR).getLiborAvg

        FundDetails = Bunch()
        FundDetails.x = self.xF
        FundDetails.type = dataType.function
        FundDetails.function = Exponential(x=self.xF).getLiborAvg
        ########################################
        myBootStrapper = BootstrapperCDSLadder(self.start, periods=self.periodsCDS, LiborFunc=self.LiborDetails, QFunc=self.QDetails,
                                               OISFunc=self.OISDetails)
        self.xQ = myBootStrapper.CalibrateCurve(self.xQ, quotesIn)
        return self.xQ

    def setDateList(self,observationDatelist):
        self.observationDatelist = observationDatelist
        self.getScheduleCF()
        self.scheduleComplete = list(set(self.observationDatelist+self.scheduleCF))
        self.refreshOIS()
        self.refreshBankOIS()
        self.refreshLibor()
        self.refreshQ()
        self.completeDatelist.sort()
        self.OIS = self.myCorp.getCorporateData(rating='OIS', datelist=self.observationDatelist)
        self.FundingCurve = self.myCorp.getCorporateData(rating='AA', datelist=self.completeDatelist)
        self.CurveStorage = self.model(x=self.xQ, trim_start=self.trim_start, trim_end=self.trim_end, simNumber=self.simNumber, t_step=self.t_step)
        self.Q = self.CurveStorage.getSmallLibor(tenors=self.observationDatelist)
        self.FVARates = (self.FundingCurve[1:]-self.FundingCurve[0:-1:])*self.FundingCurve[0:-1:]
        self.FVARates -= (self.OIS[1:]-self.OIS[0:-1:])*self.OIS[0:-1:]
        self.DeltaQ=pd.DataFrame(self.Q[0:-1:].values-self.Q[1:].values,index=self.Q.index[1:])
        self.DeltaQ.loc[self.Q.index[0]]= 0.0
        self.DeltaQ.sort_index()

    def addTrade(self,trade):
        trade.NettingNodeId = self.NettingNodeId
        trade.Counterparty = self.Counterparty
        self.trades.append(trade)
        self.getScheduleCF()

    def calcExposure(self):
        self.EPE = {}
        self.ENE = {}
        for t in self.observationDatelist:
            self.calcEPE_ENE(t)
        self.EPE = pd.DataFrame.from_dict(self.EPE, orient='index')
        self.ENE = pd.DataFrame.from_dict(self.ENE, orient='index')

    def calcEPE_ENE(self,referenceDate):
        pvs = np.zeros([1,self.simNumber])
        for x in self.trades:
            pvs += x.getPV(referenceDate)
        self.EPE[referenceDate]= np.mean(pvs*(pvs>0))
        self.ENE[referenceDate]= np.mean(pvs*(pvs<0))

    def calcCVA_DVA(self,observationDatelist):
        self.CVA = 0.0
        self.DVA = 0.0
        self.setDateList(observationDatelist)
        self.calcExposure()
        for t in self.observationDatelist:
            self.CVA += self.EPE[t]*(1-self.R)*self.OIS.loc[t]*self.DeltaQ.loc[t]
            self.DVA += self.ENE[t]*(1-self.R)*self.OIS.loc[t]*self.DeltaQ.loc[t]

    def calcFVA(self,observationDatelist):
        self.setDateList(observationDatelist)
        self.calcExposure()
        self.FVA = np.sum(0.5*(self.ENE[1:]+self.ENE[0:-1:])*self.OIS*self.Q*self.FVARates)

    def calcBenefitFVA(self,observationDatelist):
        self.setDateList(observationDatelist)
        self.calcExposure()
        self.BenefitFVA = np.sum(0.5*(self.EPE[1:]+self.EPE[0:-1:])*self.OIS*self.Q*self.FVARates)

    def getScheduleCF(self):
        self.scheduleCF = []
        for t in self.trades:
            self.scheduleCF += list(set(self.scheduleCF + t.getScheduleCF()))
        self.scheduleCF = list(set(self.scheduleCF))
        self.scheduleCF.sort()
        return self.scheduleCF


    def refreshLibor(self):
        if (self.LiborFunc.type is dataType.none):
            return
        if (self.LiborFuncIsTrue):
            lib = self.LiborFunc.function(x=self.LiborFunc.x, tenors=self.scheduleComplete)
        else:
            lib = self.LiborFunc.array

        if (len(lib) != 0):
            self.LiborReferenceDate = pd.DataFrame(np.tile(lib.loc[self.referenceDate], [np.shape(self.OIS)[0], 1]),
                                                   index=lib.index)
            self.LiborMaturity = pd.DataFrame(np.tile(lib.loc[self.maturity], [np.shape(self.OIS)[0], 1]),
                                              index=lib.index)
            self.LiborStart = pd.DataFrame(np.tile(lib.loc[self.start_date], [np.shape(self.OIS)[0], 1]),
                                           index=lib.index)
            self.Libor = lib / self.LiborReferenceDate

        if (str(self.legType).upper() == 'FLOATING'):
            if (self.inArrears):
                lib = pd.DataFrame((self.Libor[0:-1:].values - self.Libor[1:].values) / self.Libor[1:].values,
                                   index=self.Libor.index[1:])
            else:
                lib = pd.DataFrame((self.Libor[0:-1:].values - self.Libor[1:].values) / self.Libor[1:].values,
                                   index=self.Libor.index[0:-1:])
            self.rates = (lib / self.accruals).dropna()

    def refreshOIS(self):
        if (self.OISFunc.type is dataType.none):
            return
        OIS = []
        if (self.OISFuncIsTrue):
            OIS = self.OISFunc.function(x=self.OISFunc.x, tenors=self.scheduleComplete)
        else:
            OIS = self.OISFunc.array
        if (len(OIS) != 0):
            self.OISReferenceDate = pd.DataFrame(np.tile(OIS.loc[self.referenceDate], [np.shape(OIS)[0], 1]),
                                                 index=OIS.index)
            self.OISMaturity = pd.DataFrame(np.tile(OIS.loc[self.maturity], [np.shape(OIS)[0], 1]),
                                            index=OIS.index)
            self.OISStart = pd.DataFrame(np.tile(OIS.loc[self.start_date], [np.shape(OIS)[0], 1]),
                                         index=OIS.index)
            self.OIS = OIS / self.OISReferenceDate

            if (self.inArrears):
                self.avgOIS = 0.5 * pd.DataFrame(self.OIS[0:-1:].values + self.OIS[1:].values, index=self.OIS.index[1:])
            else:
                self.avgOIS = 0.5 * pd.DataFrame(self.OIS[0:-1:].values + self.OIS[1:].values,
                                                 index=self.OIS.index[0:-1:])
            self.avgOIS[self.avgOIS>1.0]=1.0

            try:
                self.principalAtMaturity = (self.referenceDate <= self.maturity)*(pd.DataFrame(self.principal * np.ones([1, self.simNumber]),
                                                        index=[self.maturity]) * self.OIS.loc[self.maturity]).dropna()
            except:
                pass
            self.fee = (self.referenceDate == self.start_date)*pd.DataFrame(self.feeValue * np.ones([1, self.simNumber]), index=[self.start_date])

    def refreshBankOIS(self):
        if (self.QBankFunc.type is dataType.none):
            return
        OIS = []
        if (self.OISFuncIsTrue):
            OIS = self.OISFunc.function(x=self.OISFunc.x, tenors=self.scheduleComplete)
        else:
            OIS = self.OISFunc.array
        if (len(OIS) != 0):
            self.OISReferenceDate = pd.DataFrame(np.tile(OIS.loc[self.referenceDate], [np.shape(OIS)[0], 1]),
                                                 index=OIS.index)
            self.OISMaturity = pd.DataFrame(np.tile(OIS.loc[self.maturity], [np.shape(OIS)[0], 1]),
                                            index=OIS.index)
            self.OISStart = pd.DataFrame(np.tile(OIS.loc[self.start_date], [np.shape(OIS)[0], 1]),
                                         index=OIS.index)
            self.OIS = OIS / self.OISReferenceDate

            if (self.inArrears):
                self.avgOIS = 0.5 * pd.DataFrame(self.OIS[0:-1:].values + self.OIS[1:].values, index=self.OIS.index[1:])
            else:
                self.avgOIS = 0.5 * pd.DataFrame(self.OIS[0:-1:].values + self.OIS[1:].values,
                                                 index=self.OIS.index[0:-1:])
            self.avgOIS[self.avgOIS>1.0]=1.0

            try:
                self.principalAtMaturity = (self.referenceDate <= self.maturity)*(pd.DataFrame(self.principal * np.ones([1, self.simNumber]),
                                                        index=[self.maturity]) * self.OIS.loc[self.maturity]).dropna()
            except:
                pass
            self.fee = (self.referenceDate == self.start_date)*pd.DataFrame(self.feeValue * np.ones([1, self.simNumber]), index=[self.start_date])

    def refreshQ(self):
        if (self.QFunc.type is dataType.none):
            return
        if (self.QFuncIsTrue):
            Q = self.QFunc.function(x=self.QFunc.x, tenors=self.scheduleComplete)
        else:
            Q = self.QFunc.array

        if (len(Q) != 0):
            self.QReferenceDate = pd.DataFrame(np.tile(Q.loc[self.referenceDate], [np.shape(Q)[0], 1]),
                                               index=Q.index)
            self.QMaturity = pd.DataFrame(np.tile(Q.loc[self.maturity], [np.shape(Q)[0], 1]),
                                          index=Q.index)
            self.QStart = pd.DataFrame(np.tile(Q.loc[self.start_date], [np.shape(Q)[0], 1]),
                                       index=Q.index)
            self.Q = Q / self.QReferenceDate

        if (str(self.legType).upper() == 'DEFAULT'):
            if (self.inArrears):
                lib = pd.DataFrame((self.Q[0:-1:].values - self.Q[1:].values) / self.Q[1:].values,
                                   index=self.Q.index[1:])
            else:
                lib = pd.DataFrame((self.Q[0:-1:].values - self.Q[1:].values) / self.Q[1:].values,
                                   index=self.Q.index[0:-1:])
            self.rates = ((1 - self.R) * lib / self.accruals).dropna()

        if (str(self.legType).upper() == 'DEFAULT_TRANCHE'):
            self.getQTranche(self.K1,self.K2)
            if (self.inArrears):
                lib = pd.DataFrame((self.QTranche[0:-1:].values - self.QTranche[1:].values) / self.QTranche[1:].values,
                                   index=self.Q.index[1:])
            else:
                lib = pd.DataFrame((self.QTranche[0:-1:].values - self.QTranche[1:].values) / self.QTranche[1:].values,
                                   index=self.QTranche.index[0:-1:])
            self.rates = ((1 - self.R) * lib / self.accruals).dropna()
