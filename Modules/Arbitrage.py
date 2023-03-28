from scipy.optimize import newton as root
import numpy as np

class referenceArbitrage:

    def __init__(self, priceProcess, CFMM):
        self.CFMM = CFMM
        self.p = priceProcess
        self.xFees = []
        self.yFees = []
        self.Volume = []

    def arbitrage(self):

        if self.p - 1e-8 > self.CFMM.marginalPrice():
            
            FeeEarned = self.CFMM.swapYforX(self.CFMM.arbAmount(self.p))[1]
            SwapVolume = self.CFMM.swapYforX(self.CFMM.arbAmount(self.p))[0]*self.p
            self.yFees.append(FeeEarned)
            self.Volume.append(SwapVolume)

        elif self.p + 1e-8 < self.CFMM.marginalPrice():

            FeeEarned = self.CFMM.swapXforY(self.CFMM.arbAmount(self.p))[1]
            SwapVolume = self.CFMM.swapXforY(self.CFMM.arbAmount(self.p))[0]
            self.xFees.append(FeeEarned)
            self.Volume.append(SwapVolume)
        
        else:
            pass
    def typeCheck(self):    
        print(self.CFMM.marginalPrice())