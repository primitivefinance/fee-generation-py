import numpy as np
from scipy.stats import norm

class RMM01:
    '''
    RMM-01 pool logic.
    Init params include:
    p0      - initial pool price
    K       - strike
    sigma   - IV parameter
    T       - pool duration
    dt      - tau update timesteps
    gamma   - fee regime
    env     - environment variable
    '''
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
        if 1 - (deltain + self.x) < 0:
            deltaout = 0
            return deltaout, 0
        else:
            x_temp = self.x + (1 - self.fee) * deltain
            deltaout = self.y - self.TradingFunction() - self.strike*norm.cdf(norm.ppf(1 - x_temp) - self.iv * np.sqrt(tau))
            if deltaout < 1e-9:
                deltaout = 0
                return deltaout, 0
            else:
                self.x += deltain
                self.y -= deltaout
                feeEarned = self.fee * deltain
                return deltaout, feeEarned

    def virtualswapXforY(self, deltain):
        tau = self.T - self.dt * self.env.now
        if 1 - (deltain + self.x) < 0:
            deltaout = 0
            return deltaout, 0
        else:
            x_temp = self.x + (1 - self.fee) * deltain
            deltaout = self.y - self.TradingFunction() - self.strike*norm.cdf(norm.ppf(1 - x_temp) - self.iv * np.sqrt(tau))
            if deltaout < 1e-9:
                deltaout = 0
                return deltaout, 0
            else:
                feeEarned = self.fee * deltain
                return deltaout, feeEarned        

    def swapYforX(self, deltain):
        tau = self.T - self.dt * self.env.now
        if self.strike + self.TradingFunction() - (deltain + self.y) < 0:
            deltaout = 0
            return deltaout, 0
        else:
            y_temp = self.y + (1 - self.fee) * deltain
            deltaout = self.x - 1 + norm.cdf(norm.ppf((y_temp - self.TradingFunction())/self.strike) + self.iv * np.sqrt(tau))
            if deltaout < 1e-9:
                deltaout = 0
                return deltaout, 0
            else:
                self.y += deltain
                self.x -= deltaout
                feeEarned = self.fee * deltain
                return deltaout, feeEarned

    def virtualswapYforX(self, deltain):
        tau = self.T - self.dt * self.env.now
        if self.strike + self.TradingFunction() - (deltain + self.y) < 0:
            deltaout = 0
            return deltaout, 0
        else:
            y_temp = self.y + (1 - self.fee) * deltain
            deltaout = self.x - 1 + norm.cdf(norm.ppf((y_temp - self.TradingFunction())/self.strike) + self.iv * np.sqrt(tau))
            if deltaout < 1e-9:
                deltaout = 0
                return deltaout, 0
            else:
                feeEarned = self.fee * deltain
                return deltaout, feeEarned        

    def arbAmount(self, s):
        if s < self.marginalPrice():
            tau = self.T - self.dt * self.env.now
            deltain = (1 - norm.cdf(np.log(s/self.strike)/(self.iv*np.sqrt(tau)) + 0.5*self.iv*np.sqrt(tau)) - self.x)/(1 - self.fee)
            return deltain

        elif s > self.marginalPrice():
            tau = self.T - self.dt * self.env.now
            x_temp = 1 - norm.cdf(np.log(s/self.strike)/(self.iv*np.sqrt(tau)) + 0.5*self.iv*np.sqrt(tau))
            deltain = (self.strike*norm.cdf(norm.ppf(1 - x_temp) - self.iv*np.sqrt(tau)) + self.TradingFunction() - self.y)/(1 - self.fee)
            return deltain

        else:
            deltain = 0
            return deltain
        
class StableVolatility:
    '''
    Static Volatility pool logic. It's essentially an RMM-01 pool with a static time to expiry.
    Init params include:
    p0      - initial pool price
    K       - strike
    sigma   - IV parameter
    T       - pool duration
    gamma   - fee regime
    env     - environment variable
    '''
    def __init__(self, p0, K, sigma, T, gamma, env, shares):
        self.p0 = p0
        self.env = env
        self.strike = K
        self.T = T
        self.iv = sigma
        self.shares = shares
        self.fee = 1 - gamma
        self.x = (1 - norm.cdf(np.log(self.p0/self.strike)/(self.iv*np.sqrt(self.T)) + 0.5*self.iv*np.sqrt(self.T)))*self.shares
        self.y = self.strike*norm.cdf(norm.ppf(1-self.x/self.shares) - self.iv*np.sqrt(self.T))*self.shares
    
    def TradingFunction(self):
        tau = self.T
        k = self.y/self.shares - self.strike*norm.cdf(norm.ppf(1 - self.x/self.shares) - self.iv*np.sqrt(tau))
        return k

    def marginalPrice(self):
        tau = self.T
        return self.strike*np.exp(norm.ppf(1 - self.x/self.shares)*self.iv*np.sqrt(tau) - 0.5*self.iv**2*tau)

    def swapXforY(self, deltain):
        tau = self.T
        if self.shares - (deltain + self.x) < 1e-9:
            deltaout = 0
            return deltaout, 0
        else:
            x_temp = self.x + (1 - self.fee) * deltain
            deltaout = self.y - self.TradingFunction()*self.shares - self.shares*self.strike*norm.cdf(norm.ppf(1 - x_temp/self.shares) - self.iv * np.sqrt(tau))
            if deltaout < 1e-9:
                deltaout = 0
                return deltaout, 0
            else:
                self.x += deltain
                self.y -= deltaout
                feeEarned = self.fee * deltain
                return deltaout, feeEarned

    def virtualswapXforY(self, deltain):
        tau = self.T
        if self.shares - (deltain + self.x) < 1e-9:
            deltaout = 0
            return deltaout, 0
        else:
            x_temp = self.x + (1 - self.fee) * deltain
            deltaout = self.y - self.shares*self.TradingFunction() - self.shares*self.strike*norm.cdf(norm.ppf(1 - x_temp/self.shares) - self.iv * np.sqrt(tau))
            if deltaout < 1e-9:
                deltaout = 0
                return deltaout, 0
            else:
                feeEarned = self.fee * deltain
                return deltaout, feeEarned
            
    def swapYforX(self, deltain):
        tau = self.T
        if (self.strike + self.TradingFunction()) * self.shares - (deltain + self.y) < 1e-9:
            deltaout = 0
            return deltaout, 0
        else:
            y_temp = self.y + (1 - self.fee) * deltain
            deltaout = self.x - self.shares + self.shares * norm.cdf(norm.ppf((y_temp/self.shares - self.TradingFunction())/self.strike) + self.iv * np.sqrt(tau))
            if deltaout < 1e-9:
                deltaout = 0
                return deltaout, 0
            else:
                self.y += deltain
                self.x -= deltaout
                feeEarned = self.fee * deltain
                return deltaout, feeEarned

    def virtualswapYforX(self, deltain):
        tau = self.T
        if (self.strike + self.TradingFunction()) * self.shares - (deltain + self.y) < 1e-9:
            deltaout = 0
            return deltaout, 0
        else:
            y_temp = self.y + (1 - self.fee) * deltain
            deltaout = self.x - self.shares + self.shares * norm.cdf(norm.ppf((y_temp/self.shares - self.TradingFunction())/self.strike) + self.iv * np.sqrt(tau))
            if deltaout < 1e-9:
                deltaout = 0
                return deltaout, 0
            else:
                feeEarned = self.fee * deltain
                return deltaout, feeEarned

    def arbAmount(self, s):
        if s < self.marginalPrice():
            tau = self.T 
            deltain = (self.shares - self.shares*norm.cdf(np.log(s/self.strike)/(self.iv*np.sqrt(tau)) + 0.5*self.iv*np.sqrt(tau)) - self.x)/(1 - self.fee)
            return deltain

        elif s > self.marginalPrice():
            tau = self.T
            x_temp = 1 - norm.cdf(np.log(s/self.strike)/(self.iv*np.sqrt(tau)) + 0.5*self.iv*np.sqrt(tau))
            deltain = (self.strike*norm.cdf(norm.ppf(1 - x_temp) - self.iv*np.sqrt(tau))*self.shares + self.TradingFunction()*self.shares - self.y)/(1 - self.fee)
            return deltain

        else:
            deltain = 0
            return deltain
        
class ConstantSum:
    def __init__(self, K, init_x, init_y, gamma, env):
        self.K = K
        self.env = env
        self.x = init_x
        self.y = init_y
        self.gamma = gamma

    def TradingFunction(self):
        return self.x * self.K + self.y

    def marginalPrice(self):
        return self.K

    def swapXforY(self, deltain):

        deltaout = self.K * deltain * self.gamma
        self.x += deltain
        self.y -= deltaout
        FeeEarned = (1 - self.gamma) * deltain
        return deltaout, FeeEarned
    
    def swapYforX(self, deltain):

        deltaout = deltain * self.gamma / self.K
        self.y += deltain
        self.x -= deltaout
        FeeEarned = (1 - self.gamma) * deltain
        return deltaout, FeeEarned
    
    def virtualswapXforY(self, deltain):

        deltaout = self.K * deltain * self.gamma
        FeeEarned = (1 - self.gamma) * deltain
        return deltaout, FeeEarned
    
    def virtualswapYforX(self, deltain):
        
        deltaout = deltain * self.gamma / self.K
        FeeEarned = (1 - self.gamma) * deltain
        return deltaout, FeeEarned
    
    def arbAmount(self, s):

        if s < self.K:
            deltaout = self.y
            deltain = deltaout / (self.gamma * self.K)
            return deltain
        elif s > self.K:
            deltaout = self.x
            deltain = deltaout * self.gamma * self.K
            return deltain
        else:
            deltain = 0
            return deltain