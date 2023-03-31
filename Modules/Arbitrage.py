from scipy.optimize import newton as root
import numpy as np

class referenceArbitrage:

    def __init__(self, priceProcess, CFMM):
        self.CFMM = CFMM
        self.p = priceProcess
        self.Fees = 0
        self.Volume = 0

    def arbitrage(self):

        if self.p - self.CFMM.marginalPrice() > 1e-9:
            
            Swap = self.CFMM.swapYforX(self.CFMM.arbAmount(self.p))
            self.Fees = Swap[1]
            self.Volume = Swap[0]*self.p

        elif self.CFMM.marginalPrice() - self.p > 1e-9:

            Swap = self.CFMM.swapXforY(self.CFMM.arbAmount(self.p))
            self.Fees = Swap[1]*self.p
            self.Volume = Swap[0]
        
        else:
            pass