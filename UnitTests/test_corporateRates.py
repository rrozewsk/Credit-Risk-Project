from datetime import date
from unittest import TestCase

import numpy as np
import pandas as pd
from Curves.Corporates.CorporateDaily import CorporateRates


from parameters import WORKING_DIR

periods = '1Y'
freq = '1M'
t_step = 1.0 / 365.0
simNumber = 10
start = date(2005, 3, 30)
trim_start = date(2000, 1, 1)
trim_end = date(2010, 12, 31)
referenceDate = date(2005, 3, 30)
xR = [2.0, 0.05, 0.01, 0.07]

# CashFlow Dates
myCorp = CorporateRates()
myCorp.getCorporates(trim_start, trim_end)
myCorp.pickleMe()

scheduleComplete = pd.date_range(start=trim_start,end=trim_end)


class TestCorporateRates(TestCase):
    def test_getOIS(self):
        OIS = myCorp.getOISData(datelist=scheduleComplete)
        print('AAA', np.shape(OIS))

    def test_getCorporateData1(self):
        AAA = myCorp.getCorporateData(rating='AAA')
        print(np.shape(AAA))

    def test_getCorporateData2(self):
        OIS = myCorp.getCorporateData(rating='OIS', datelist=scheduleComplete)
        print('OIS', np.shape(OIS))

    def test_pickleMe(self):
        return
        fileName = WORKING_DIR + '/myCorp'
        myCorp.pickleMe(fileName)

    def test_unPickleMe(self):
        fileName = WORKING_DIR + '/myCorp.dat'
        myCorp.unPickleMe(fileName)

    def test_saveMeExcel(self):
        fileName = WORKING_DIR + '/myCorp.xlsx'
        myCorp.saveMeExcel(whichdata=myCorp.corporates, fileName=fileName)
