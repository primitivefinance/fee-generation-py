import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import simpy

from Arbitrage import referenceArbitrage as a
from CFMM import RMM01
from CFMM import StableVolatility
from CFMM import ConstantSum
import PriceGen as price
from PriceGen import close_values

# Simulation Choices

run_GBM_simulation = False
run_OU_simulation = True
run_Backtest = False
run_ConstantSum_test = False

# Config Params

G = 100      # Number of Pool Realized Volatility Values

K1 = 1500                               # Strike of RMM-01 Pool
p0 = 1500                               # Initial RMM01 and GBM Price
K2 = 1                                  # Strike of StableVolatility Pool
P0 = 1                                  # Initial OU and StableVolatility Price
sigma = np.linspace(0.0001, 0.1, G)     # Pool Implied Volatility Parameter
T = 0.1                                 # Pool Time Horizon in Years
dt = 0.015/365                          # Time-Step Size in Years
N = round(T/dt)                         # Number of Time-Steps
gamma = 0.9999                          # Fee Regime on CFMM
shares = 100000                         # Number of Shares in CFMM

v = 0.1             # GBM & OU Volatility Parameter
mu = 0.0            # GBM Drift Parameter

mean = 1            # OU mean price
theta = 2/365       # OU mean reversion time

Arb = 5             # Arbitrage Profit Threshold Denominated in Numeraire

M = 1               # Number of Simulation Runs per IV parameter

# Simulation Processes

def simulateGBM(env, i, Fees):
    CFMM = RMM01(p0, K1, sigma[i], T, dt, gamma, env, shares)
    
    while True:

        GBM = price.generateGBM(T, mu, v, p0, dt, env)
        arb = a(GBM, CFMM, Arb)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

def simulateOU(env, i, Fees):
    CFMM = StableVolatility(P0, K2, sigma[i], T, gamma, env, shares)

    while True:

        OU = price.generateOU(T, mean, v, P0, dt, theta, env)
        arb = a(OU, CFMM, Arb)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

def simulateBacktest(env, i, Fees):
    CFMM = StableVolatility(P0, K2, sigma[i], T, gamma, env, shares)

    while True:

        arb = a(close_values[env.now], CFMM, Arb)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

def simulateConstantSum(env):
    CFMM = ConstantSum(K2, shares/2, shares/2, gamma, env)

    while True:
        OU = price.generateOU(T, mean, v, P0, dt, theta, env)
        arb = a(OU, CFMM, Arb)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

# GBM Based RMM01 simulation

def GBMSimProcess(j):
    FeeIncome = []
    for i in range (0, M):
        Fees = []
        env = simpy.Environment()
        env.process(simulateGBM(env, j, Fees))
        env.run(until=N)

        FeeIncome.append(sum(Fees))
    return sum(FeeIncome)/M

## Multithreading

array = []
if run_GBM_simulation:
    with ThreadPoolExecutor() as executor:
        array = list(executor.map(GBMSimProcess, range(G)))

# OU Based Stable Volatility simulation

def OUSimProcess(j):
    FeeIncome = []
    for i in range (0, M):
        Fees = []
        env = simpy.Environment()
        env.process(simulateOU(env, j, Fees))
        env.run(until=N)

        FeeIncome.append(sum(Fees))
    return sum(FeeIncome)/M

## Multithreading

array2 = []
if run_OU_simulation:
    with ThreadPoolExecutor() as executor:
        array2 = list(executor.map(OUSimProcess, range(G)))

# Backtest Stable Volatility simulation

def BacktestProcess(j):
    FeeIncome = []
    Fees = []
    env = simpy.Environment()
    env.process(simulateBacktest(env, j, Fees))
    env.run(until=len(price.close_values))

    FeeIncome.append(sum(Fees))
    return sum(FeeIncome)

## Multithreading

array3 = []
if run_Backtest:
    with ThreadPoolExecutor() as executor:
        array3 = list(executor.map(BacktestProcess, range(G)))

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
    plt.title(f"Strike {K1}, Time Horizon = {T} years, Fee = {(1-gamma)*100}%, RV = {v*100}% annualized, Drift = {mu*100}%, GBM RMM-01 Simulation", fontsize=10) 
    plt.xlabel("Implied Volatility Parameter", fontsize=10)
    plt.ylabel("Expected Fees", fontsize=10)
    plt.show()
elif run_OU_simulation:
    plt.plot(sigma, array2, 'g-')
    plt.title(f"Strike {K2}, Time Horizon = {T} years, Fee = {(1-gamma)*100}%, RV = {v*100}% annualized, Mean Price = {mean}, Theta = {theta}, OU Stable Volatility Simulation", fontsize=10)
    plt.xlabel("Implied Volatility Parameter", fontsize=10)
    plt.ylabel("Expected Fees", fontsize=10)
    plt.show()
elif run_Backtest:
    plt.plot(sigma, array3, 'g-')
    plt.title(f"Strike {K2}, Time Horizon = {T} years, Fee = {(1-gamma)*100}%, RV = 2.56% annualized, Backtest USDC/USDT", fontsize=10)
    plt.xlabel("Implied Volatility Parameter", fontsize=10)
    plt.ylabel("Expected Fees", fontsize=10)
    plt.show()