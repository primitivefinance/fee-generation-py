import numpy as np

def generateGBM(T, mu, sigma, p0, dt, env):
            N = round(T/dt)
            t = dt*env.now
            dW = np.random.standard_normal(size=N)
            W = np.cumsum(dW)[int(env.now)]*np.sqrt(dt)
            P = p0*np.exp((mu - 0.5*sigma**2)*t + sigma*W)
            return P

def generateOU(T, mean, sigma, p0, dt, theta, env):
    N = round(T/dt)
    dW = np.random.standard_normal(size=N)
    W = np.cumsum(dW)[int(env.now)] * np.sqrt(dt)
    t = dt * env.now
    term = np.exp(-theta * t)
    OU = p0 * term + mean * (1 - term) + sigma * np.sqrt(dt / (2 * theta)) * W
    return OU
      
