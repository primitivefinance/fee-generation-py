import numpy as np
from scipy.stats import norm

class RMM01:

    def __init__(self, p0, K, sigma, T, dt, gamma, env):
        self.p0 = p0
        self.env = env
        self.strike = K
        self.iv = sigma
        self.tau = T - dt*self.env.now
        self.k = 0
        self.fee = 1 - gamma
        self.x = norm.cdf(-np.log(self.p0/self.strike)/(self.iv*np.sqrt(self.tau)) - 0.5*self.iv*self.tau)
        self.y = self.strike*norm.cdf(norm.ppf(1-self.x) - self.iv*np.sqrt(self.tau)) + self.k

    def marginalPrice(self):
        return self.strike*np.exp(norm.ppf(1 - self.x)*self.iv*np.sqrt(self.tau) - 0.5*self.iv**2*self.tau)

    def swapXforY(self, deltain):
        assert deltain >= 0
        x_temp = self.x + (1 - self.fee)*deltain
        deltaout = self.y - self.k - self.strike*norm.cdf(norm.ppf(1 - x_temp) - self.iv*np.sqrt(self.tau))
        self.x += deltain
        self.y -= deltaout
        self.k = self.y - self.strike*norm.cdf(norm.ppf(1 - self.x) - self.iv*np.sqrt(self.tau))
        feeEarned = self.fee*deltain
        return deltaout, feeEarned

    def swapYforX(self, deltain):
        assert deltain >= 0
        y_temp = self.y + (1 - self.fee)*deltain
        deltaout = self.x - 1 + norm.cdf(norm.ppf((self.y + (1 - self.fee)*deltain - self.k)/self.strike) + self.iv*np.sqrt(self.tau))
        self.y += deltain
        self.x -= deltaout
        self.k = self.y - self.strike*norm.cdf(norm.ppf(1 - self.x) - self.iv*np.sqrt(self.tau))
        feeEarned = self.fee*deltain
        return deltaout, feeEarned

    def arbAmount(self, s):
        if s < self.marginalPrice():
            deltain = (1 - norm.cdf(np.log(s/self.strike)/(self.iv*np.sqrt(self.tau)) + 0.5*self.iv*np.sqrt(self.tau)) - self.x)/(1 - self.fee)
            return deltain

        elif s > self.marginalPrice():
            x_temp = 1 - norm.cdf(np.log(s/self.strike)/(self.iv*np.sqrt(self.tau)) + 0.5*self.iv*np.sqrt(self.tau))
            deltain = (self.strike*norm.cdf(norm.ppf(1 - x_temp) - self.iv*np.sqrt(self.tau)) + self.k - self.y)/(1 - self.fee)
            return deltain

        else:
            deltain = 0
            return deltain