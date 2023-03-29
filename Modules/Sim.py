import matplotlib.pyplot as plt
import numpy as np
from Arbitrage import referenceArbitrage as a
from CFMM import RMM01
from CFMM import StableVolatility
import PriceGen as price
import simpy

# Config Params

K = 1               # Strike of RMM-01 Pool
p0 = 1500           # Initial Pool and GBM Price
v = 1.1             # Implied Volatility RMM-01 Parameter
T = 7/365           # Pool Duration in Years
dt = 0.015/365      # Time-Step Size in Years
N = round(T/dt)     # Number of Time-Steps
gamma = 0.997       # Fee Regime on CFMM

mu = 0.0                        # GBM Drift Parameter
sigma = np.linspace(0.01,3,100) # GBM Realized Volatility Parameter

P0 = 1              # OU start price
mean = 1            # OU mean price
theta = 2/365       # OU mean reversion time

M = 10              # Number of Simulation Runs per RV parameter

# Simulation Processes

def simulateGBM(env, i):
    CFMM = RMM01(p0, K, sigma[i], T, dt, gamma, env)
    
    while True:

        GBM = price.generateGBM(T, mu, v, p0, dt, env)
        arb = a(GBM, CFMM)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

def simulateOU(env, i):
    CFMM = StableVolatility(P0, K, sigma[i], T, gamma, env)

    while True:

        OU = price.generateOU(T, mean, v, P0, dt, theta, env)
        arb = a(OU, CFMM)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

# GBM Based RMM01 simulation

def simulationEngineGBM():
    array = []
    for j in range (0, 50):
      
        FeeIncome = []        
        for i in range (0, M):
            Fees = []
            env = simpy.Environment()
            env.process(simulateGBM(env, j))
            env.run(until=N)
    
            FeeIncome.append(sum(Fees))
        array.append(sum(FeeIncome)/M)
    return array

# OU Based Stable Volatility simulation

def simulationEngineOU():
    array2 = []
    for j in range (0, 100):
        FeeIncome = []
        for i in range (0, M):
            Fees = []
            env = simpy.Environment()
            env.process(simulateOU(env, j))
            env.run(until=N)

            FeeIncome.append(sum(Fees))
        array2.append(sum(FeeIncome)/M)
    return array2

# Plotting Implied Volatility Parameter vs. Average Fees Generated over M OUs of static RV

plt.plot(sigma, simulationEngineOU(), 'g-')
plt.xlabel("Pool Implied Volatility", fontsize=12)
plt.ylabel("Expected Fees", fontsize=12)
plt.title("Strike 1, Initial Price 2, RV = 0.7, T = 1 week, fee = 3%")
plt.show()