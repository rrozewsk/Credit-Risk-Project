from unittest import TestCase

from parameters import xR, simNumber, t_step, trim_start, trim_end, freq, periods, referenceDate, start, inArrears, WORKING_DIR
from MonteCarloSimulators.Vasicek.vasicekMCSim import  MC_Vasicek_Sim
from Scheduler.Scheduler import Scheduler


mySchedule = Scheduler()

# Global Variables
mySchedule = mySchedule.getSchedule(start=trim_start,end=trim_end, freq="M")
mySimulator = MC_Vasicek_Sim(datelist=mySchedule,x=xR, simNumber=simNumber,t_step=t_step)


class TestMC_Vasicek_Sim(TestCase):
    def test_getLibor(self):
        mySimulator.getLibor()

    def test_getSmallLibor(self):
        # Monte Carlo trajectories creation - R
        Libor = mySimulator.getSmallLibor(mySchedule)
        return Libor

