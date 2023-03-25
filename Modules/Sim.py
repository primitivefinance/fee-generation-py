import matplotlib.pyplot as plt
import numpy as np
from Arbitrage import referenceArbitrage as a
from CFMM import RMM01
import simpy

# Config

K = 2000
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

def simulate(env):
    while True:
        CFMM = RMM01(p0, K, v, T, dt, gamma, env)

        def generateGBM(T, mu, sigma, p0, dt, env):
            N = round(T/dt)
            t = T/dt*env.now
            dW = np.random.standard_normal()
            W = np.cumsum(dW)*np.sqrt(dt)
            P = p0*np.exp((mu - 0.5*sigma**2)*t + sigma*W)
            return P

        arb = a(generateGBM(T, mu, sigma, p0, dt, env), CFMM)
        arb.arbitrage()
        print(arb.xFees)

env.process(simulate(env))
env.run(until=N)
