import matplotlib.pyplot as plt
import numpy as np
from Arbitrage import referenceArbitrage as a
from CFMM import RMM01
import simpy

# Config Params

K = 1500            # Strike of RMM-01 Pool
p0 = 1500           # Initial Pool and GBM Price
v = 0.7             # Implied Volatility RMM-01 Parameter
T = 7/365           # Pool Duration in Years
dt = 0.015/365      # Time-Step Size in Years
N = round(T/dt)     # Number of Time-Steps
gamma = 0.997       # Fee Regime on CFMM

mu = 0.0                            # GBM Drift Parameter
sigma = np.linspace(0.01, 2, 50)    # GBM Realized Volatility Parameter
             
M = 10              # Number of Simulation Runs per RV parameter

# Simulation Process

def generateGBM(T, mu, sigma, p0, dt, env):
            N = round(T/dt)
            t = dt/T*env.now
            dW = np.random.standard_normal(size=N)
            W = np.cumsum(dW)[env.now]*np.sqrt(dt)
            P = p0*np.exp((mu - 0.5*sigma**2)*t + sigma*W)
            return P

def simulate(env, i):
    CFMM = RMM01(p0, K, v, T, dt, gamma, env)
    
    while True:

        GBM = generateGBM(T, mu, sigma[i], p0, dt, env)
        arb = a(GBM, CFMM)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

# Loop simulation for each sigma

array = []
for j in range (0, 50):
      
    FeeIncome = []        
    for i in range (0, M):
        Fees = []
        env = simpy.Environment()
        env.process(simulate(env, j))
        env.run(until=N)
    
        FeeIncome.append(sum(Fees))
    array.append(sum(FeeIncome)/M)

# Plotting Realized Volatility Parameter vs. Average Fees Generated over M GBMs

plt.plot(sigma, array, 'g-')
plt.xlabel("Realized Volatility", fontsize=12)
plt.ylabel("Expected Fees", fontsize=12)
plt.title("Strike 1500, Initial Price 1500, IV = 0.7, T = 1 week, fee = 3%")
plt.show()