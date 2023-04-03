import matplotlib.pyplot as plt
import numpy as np
from Arbitrage import referenceArbitrage as a
from CFMM import RMM01
from CFMM import StableVolatility
from CFMM import ConstantSum
import PriceGen as price
from PriceGen import close_values
import simpy

# Simulation Choices

run_GBM_simulation = False
run_OU_simulation = False
run_Backtest = False
run_ConstantSum_test = True

# Config Params

K = 1               # Strike of RMM-01 Pool
p0 = 1500           # Initial RMM01 and GBM Price
v = 0.1             # Implied Volatility RMM-01 Parameter
T = 7/365           # Pool Duration in Years
dt = 0.015/365      # Time-Step Size in Years
N = round(T/dt)     # Number of Time-Steps
gamma = 0.9995      # Fee Regime on CFMM
c = 0.0025          # StableVolatility sigma*T parameter
shares = 100000     # Number of Shares in StableVolatility Pool

G = 100                             # Number of Pool Realized Volatility Values
mu = 0.0                            # GBM Drift Parameter
sigma = np.linspace(0.001, 0.1, G)  # GBM Realized Volatility Parameter

P0 = 1              # OU start price and StableVolatility initial price
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
    CFMM = StableVolatility(P0, K, sigma[i], T, gamma, env, shares)

    while True:

        OU = price.generateOU(T, mean, v, P0, dt, theta, env)
        arb = a(OU, CFMM)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

def simulateBacktest(env, i):
    CFMM = StableVolatility(P0, K, sigma[i], T, gamma, env, shares)

    while True:

        arb = a(close_values[env.now], CFMM)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

def simulateConstantSum(env):
    CFMM = ConstantSum(K, 10000, 10000, gamma, env)

    while True:
        OU = price.generateOU(T, mean, v, P0, dt, theta, env)
        arb = a(OU, CFMM)
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

# Constant Sum OU Test

if run_ConstantSum_test:
    Fees = []
    env = simpy.Environment()
    env.process(simulateConstantSum(env))
    env.run(until=N)
    FeeIncome = sum(Fees)
    print(FeeIncome)

# Plotting Implied Volatility Parameter vs. Average Fees Generated

if run_GBM_simulation:
    plt.plot(sigma, array, 'g-')
    plt.title(f"Strike {K}, Time Horizon = {T} years, Fee = {(1-gamma)*100}%, RV = {v*100}% annualized, Drift = {mu*100}%, GBM RMM-01 Simulation", fontsize=10) 
    plt.xlabel("Implied Volatility Parameter", fontsize=10)
    plt.ylabel("Expected Fees", fontsize=10)
    plt.show()
elif run_OU_simulation:
    plt.plot(sigma, array2, 'g-')
    plt.title(f"Strike {K}, Time Horizon = {T} years, Fee = {(1-gamma)*100}%, RV = {v*100}% annualized, Mean Price = {mean}, Theta = {theta}, OU Stable Volatility Simulation", fontsize=10)
    plt.xlabel("Implied Volatility Parameter", fontsize=10)
    plt.ylabel("Expected Fees", fontsize=10)
    plt.show()
elif run_Backtest:
    plt.plot(sigma, array3, 'g-')
    plt.title(f"Strike {K}, Time Horizon = {T} years, Fee = {(1-gamma)*100}%, RV = 2.56% annualized, Backtest USDC/USDT", fontsize=10)
    plt.xlabel("Implied Volatility Parameter", fontsize=10)
    plt.ylabel("Expected Fees", fontsize=10)
    plt.show()