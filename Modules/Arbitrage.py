class referenceArbitrage:

    def __init__(self, priceProcess, CFMM):
        self.CFMM = CFMM
        self.p = priceProcess
        self.Fees = 0
        self.Volume = 0

    def arbitrage(self):

        if self.p - self.CFMM.marginalPrice() > 1e-9:
            
            Swap = self.CFMM.virtualswapYforX(self.CFMM.arbAmount(self.p))

            if Swap[0] * self.p - self.CFMM.arbAmount(self.p) > 10:
                self.Fees = Swap[1]
                self.Volume = Swap[0]*self.p
                self.CFMM.swapYforX(self.CFMM.arbAmount(self.p))
            else:
                self.Fees = 0

        elif self.CFMM.marginalPrice() - self.p > 1e-9:

            Swap = self.CFMM.virtualswapXforY(self.CFMM.arbAmount(self.p))

            if Swap[0] - self.CFMM.arbAmount(self.p) * self.p > 10:
                self.Fees = Swap[1]*self.p
                self.Volume = Swap[0]
                self.CFMM.swapXforY(self.CFMM.arbAmount(self.p))
            else:
                self.Fees = 0
        
        else:
            pass