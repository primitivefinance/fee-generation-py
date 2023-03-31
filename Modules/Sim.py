import matplotlib.pyplot as plt
import numpy as np
from Arbitrage import referenceArbitrage as a
from CFMM import RMM01
from CFMM import StableVolatility
import PriceGen as price
from PriceGen import close_values
import simpy

# Simulation Choices

run_GBM_simulation = False
run_OU_simulation = False
run_Backtest = True

# Config Params

K = 1               # Strike of RMM-01 Pool
p0 = 1500           # Initial Pool and GBM Price
v = 1.1             # Implied Volatility RMM-01 Parameter
T = 0.1           # Pool Duration in Years
dt = 0.015/365      # Time-Step Size in Years
N = round(T/dt)     # Number of Time-Steps
gamma = 0.997       # Fee Regime on CFMM
c = 0.0025          # StableVolatility sigma*T parameter

G = 100                         # Number of Pool Realized Volatility Values
mu = 0.0                        # GBM Drift Parameter
sigma = np.linspace(0.01, 3, G) # GBM Realized Volatility Parameter

P0 = 1              # OU start price
mean = 1            # OU mean price
theta = 2/365       # OU mean reversion time


M = 1              # Number of Simulation Runs per RV parameter

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

def simulateBacktest(env, i):
    CFMM = StableVolatility(P0, K, sigma[i], T, gamma, env)

    while True:

        arb = a(close_values[env.now], CFMM)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

# GBM Based RMM01 simulation

array = []
if run_GBM_simulation:
    for j in range (0, G):     
        FeeIncome = []        
        for i in range (0, M):
            Fees = []
            env = simpy.Environment()
            env.process(simulateGBM(env, j))
            env.run(until=N)
        
            FeeIncome.append(sum(Fees))
        array.append(sum(FeeIncome)/M)

# OU Based Stable Volatility simulation

array2 = []
if run_OU_simulation:
    for j in range (0, G):
        FeeIncome = []
        for i in range (0, M):
            Fees = []
            env = simpy.Environment()
            env.process(simulateOU(env, j))
            env.run(until=N)

            FeeIncome.append(sum(Fees))
        array2.append(sum(FeeIncome)/M)

# Backtest Stable Volatility simulation

array3 = []
if run_Backtest:
    for j in range (0, G):
        FeeIncome = []
        Fees = []
        env = simpy.Environment()
        env.process(simulateBacktest(env, j))
        env.run(until=len(price.close_values))

        FeeIncome.append(sum(Fees))
        array3.append(sum(FeeIncome))

# Plotting Implied Volatility Parameter vs. Average Fees Generated over M OUs of static RV

if run_GBM_simulation:
    plt.plot(sigma, array, 'g-') 
elif run_OU_simulation:
    plt.plot(sigma, array2, 'g-')
elif run_Backtest:
    plt.plot(sigma, array3, 'g-')

plt.xlabel("Pool Implied Volatility", fontsize=12)
plt.ylabel("Expected Fees", fontsize=12)
plt.title("Strike 1, Initial Price 1, RV = 2.5% annualized, T = 0.1 years, fee = 0.3%, Backtest USDC/USDT")
plt.show()
