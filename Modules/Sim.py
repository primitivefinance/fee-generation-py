import matplotlib.pyplot as plt
import numpy as np
from Arbitrage import referenceArbitrage as a
from CFMM import RMM01
import simpy

# Config

K = 1500
p0 = 1500
v = 0.7
T = 7/365
dt = 0.015/365
N = round(T/dt)

mu = 0.0
sigma = 0.4
gamma = 0.997

# Simulation

env = simpy.Environment()

def generateGBM(T, mu, sigma, p0, dt, env):
            N = round(T/dt)
            t = dt/T*env.now
            dW = np.random.standard_normal(size=N)
            W = np.cumsum(dW)[env.now]*np.sqrt(dt)
            P = p0*np.exp((mu - 0.5*sigma**2)*t + sigma*W)
            return P

def simulate(env):
    CFMM = RMM01(p0, K, v, T, dt, gamma, env)
    
    while True:

        GBM = generateGBM(T, mu, sigma, p0, dt, env)
        arb = a(GBM, CFMM)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

FeeIncome = []        
for i in range (0, 100):
    Fees = []
    env.process(simulate(env))
    
env.run(until=N)
FeeIncome.append(sum(Fees))

print(sum(FeeIncome)/100)