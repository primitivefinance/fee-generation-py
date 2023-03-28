import numpy as np
from scipy.stats import norm

class RMM01:

    def __init__(self, p0, K, sigma, T, dt, gamma, env):
        self.p0 = p0
        self.env = env
        self.strike = K
        self.T = T
        self.dt = dt
        self.iv = sigma
        self.fee = 1 - gamma
        self.x = 1 - norm.cdf(np.log(self.p0/self.strike)/(self.iv*np.sqrt(self.T)) + 0.5*self.iv*np.sqrt(self.T))
        self.y = self.strike*norm.cdf(norm.ppf(1-self.x) - self.iv*np.sqrt(self.T))
    
    def TradingFunction(self):
        tau = self.T - self.dt * self.env.now
        k = self.y - self.strike*norm.cdf(norm.ppf(1 - self.x) - self.iv*np.sqrt(tau))
        return k

    def marginalPrice(self):
        tau = self.T - self.dt * self.env.now
        return self.strike*np.exp(norm.ppf(1 - self.x)*self.iv*np.sqrt(tau) - 0.5*self.iv**2*tau)

    def swapXforY(self, deltain):
        tau = self.T - self.dt * self.env.now
        if 1 - (deltain + self.x) < 1e-8:
            deltaout = 0
            return deltaout, 0
        else:
            x_temp = self.x + (1 - self.fee) * deltain
            deltaout = self.y - self.TradingFunction() - self.strike*norm.cdf(norm.ppf(1 - x_temp) - self.iv * np.sqrt(tau))
            if deltaout < 1e-8:
                deltaout = 0
                return deltaout, 0
            else:
                self.x += deltain
                self.y -= deltaout
                feeEarned = self.fee * deltain
                return deltaout, feeEarned


    def swapYforX(self, deltain):
        tau = self.T - self.dt * self.env.now
        if self.strike + self.TradingFunction() - (deltain + self.y) < 1e-8:
            deltaout = 0
            return deltaout, 0
        else:
            y_temp = self.y + (1 - self.fee) * deltain
            deltaout = self.x - 1 + norm.cdf(norm.ppf((y_temp - self.TradingFunction())/self.strike) + self.iv * np.sqrt(tau))
            if deltaout < 1e-8:
                deltaout = 0
                return deltaout, 0
            else:
                self.y += deltain
                self.x -= deltaout
                feeEarned = self.fee * deltain
                return deltaout, feeEarned

    def arbAmount(self, s):
        if s < self.marginalPrice():
            tau = self.T - self.dt * self.env.now
            deltain = (1 - norm.cdf(np.log(s/self.strike)/(self.iv*np.sqrt(tau)) + 0.5*self.iv*np.sqrt(tau)) - self.x)
            return deltain

        elif s > self.marginalPrice():
            tau = self.T - self.dt * self.env.now
            x_temp = 1 - norm.cdf(np.log(s/self.strike)/(self.iv*np.sqrt(tau)) + 0.5*self.iv*np.sqrt(tau))
            deltain = (self.strike*norm.cdf(norm.ppf(1 - x_temp) - self.iv*np.sqrt(tau)) + self.TradingFunction() - self.y)
            return deltain

        else:
            deltain = 0
            return deltain