from scipy.optimize import newton as root
import numpy as np

class referenceArbitrage:

    def __init__(self, priceProcess, CFMM):
        self.CFMM = CFMM
        self.p = priceProcess
        self.xFees = []
        self.yFees = []

    def arbitrage(self):

        if self.p > self.CFMM.marginalPrice():
            
            FeeEarned = self.CFMM.swapYforX(self.CFMM.arbAmount(self.p))[1]
            SwapVolume = self.CFMM.swapYforX(self.CFMM.arbAmount(self.p))[0]*self.p
            self.xFees.append(FeeEarned)

        elif self.p < self.CFMM.marginalPrice():

            FeeEarned = self.CFMM.swapXforY(self.CFMM.arbAmount(self.p))[1]
            SwapVolume = self.CFMM.swapXforY(self.CFMM.arbAmount(self.p))[0]
            self.yFees.append(FeeEarned)
        
        else:
         pass
    def typeCheck(self):    
        print(type(self.CFMM.marginalPrice()))