from parameters import freq, Coupon
__author__ = 'ryan rozewski'
class CDS(object):
    def __init__(self,start,maturity,freq,coupon,refDate):
        self.startDate=start
        self.end=maturity
        self.freq=freq
        self.coupon=coupon
        self.refDate=refDate

    def