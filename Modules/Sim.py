import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import simpy
import configparser

from Arbitrage import referenceArbitrage as a
import CFMM
import PriceGen as price
from cli import parse_arguments

# Utilities for Config and CLI

args = parse_arguments()

def read_config_file(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

config = read_config_file("config.ini")

# Simulation Choices

run_GBM_simulation = args.GBM
run_OU_simulation = args.OU
run_Backtest = args.Backtest
run_ConstantSum_test = args.CS

# Config Params

G = int(config["Sim"]["G"])
sigma_low = float(config["Sim"]["sigma_low"])
sigma_high = float(config["Sim"]["sigma_high"])
K1 = float(config["Pool"]["strike_RMM01"])
p0 = float(config["Pool"]["P0_RMM01"])
K2 = float(config["Pool"]["strike_SV"])
P0 = float(config["Pool"]["P0_SV"])
sigma = np.linspace(sigma_low, sigma_high, G) 
T = float(config["Pool"]["time_horizon"])
dt = float(config["Pool"]["timestep_size"])
N = round(T/dt)
gamma = float(config["Pool"]["gamma"])
shares = float(config["Pool"]["shares"])
v = float(config["GBM"]["volatility"])
mu = float(config["GBM"]["drift"])
mean = float(config["OU"]["mean"])
theta = float(config["OU"]["theta"])
M = int(config["Sim"]["M"])
Arb = float(config["Sim"]["Arb"])

# Simulation Processes

def simulateGBM(env, i, Fees):
    Curve = CFMM.RMM01(p0, K1, sigma[i], T, dt, gamma, env, shares)
    
    while True:

        GBM = price.generateGBM(T, mu, v, p0, dt, env)
        arb = a(GBM, Curve, Arb)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

def simulateOU(env, i, Fees):
    Curve = CFMM.StableVolatility(P0, K2, sigma[i], T, gamma, env, shares)

    while True:

        OU = price.generateOU(T, mean, v, P0, dt, theta, env)
        arb = a(OU, Curve, Arb)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

def simulateBacktest(env, i, Fees):
    Curve = CFMM.StableVolatility(P0, K2, sigma[i], T, gamma, env, shares)

    while True:

        arb = a(price.close_values[env.now], Curve, Arb)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

def simulateConstantSum(env):
    Curve = CFMM.ConstantSum(K2, shares/2, shares/2, gamma, env)

    while True:
        OU = price.generateOU(T, mean, v, P0, dt, theta, env)
        arb = a(OU, Curve, Arb)
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
    avgIncome = sum(FeeIncome)/M
    varIncome = sum([(fee - avgIncome)**2 for fee in FeeIncome])/M
    return avgIncome, varIncome

## Multithreading

avgIncomeGBM = []
varIncomeGBM = []
if run_GBM_simulation:
    array = []
    with ThreadPoolExecutor() as executor:
        array = list(executor.map(GBMSimProcess, range(G)))
    avgIncome, varIncome = zip(*array)

stdIncomeGBM = [np.sqrt(var) for var in varIncomeGBM]

# OU Based Stable Volatility simulation

def OUSimProcess(j):
    FeeIncome = []
    for i in range (0, M):
        Fees = []
        env = simpy.Environment()
        env.process(simulateOU(env, j, Fees))
        env.run(until=N)

        FeeIncome.append(sum(Fees))
    avgIncome = sum(FeeIncome)/M
    varIncome = sum([(fee - avgIncome)**2 for fee in FeeIncome])/M
    return avgIncome, varIncome

## Multithreading

avgIncomeOU = []
varIncomeOU = []
if run_OU_simulation:
    array = []
    with ThreadPoolExecutor() as executor:
        array = list(executor.map(OUSimProcess, range(G)))
    avgIncomeOU, varIncomeOU = zip(*array)

stdIncomeOU = [np.sqrt(var) for var in varIncomeOU]

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

IncomeBacktest = []
if run_Backtest:
    with ThreadPoolExecutor() as executor:
        IncomeBacktest = list(executor.map(BacktestProcess, range(G)))

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
    plt.plot(sigma, avgIncome, color='#2BBA58')
    plt.errorbar(sigma, avgIncomeGBM, yerr=stdIncomeGBM, ecolor='#BCF2CD', capthick=2)
    plt.title(f"Strike {K1}, Time Horizon = {T} years, Fee = {(1-gamma)*100}%, RV = {v*100}% annualized, Drift = {mu*100}%, GBM RMM-01 Simulation", fontsize=10) 
    plt.xlabel("Implied Volatility Parameter", fontsize=10)
    plt.ylabel("Expected Fees", fontsize=10)
    plt.show()
    plt.close()

elif run_OU_simulation:
    plt.plot(sigma, avgIncomeOU, color='#2BBA58')
    plt.errorbar(sigma, avgIncomeOU, yerr=stdIncomeOU, ecolor='#BCF2CD', capthick=2)
    plt.title(f"Strike {K2}, Time Horizon = {T} years, Fee = {(1-gamma)*100}%, RV = {v*100}% annualized, Mean Price = {mean}, Theta = {theta}, OU Stable Volatility Simulation", fontsize=10)
    plt.xlabel("Implied Volatility Parameter", fontsize=10)
    plt.ylabel("Expected Fees", fontsize=10)
    plt.show()
    plt.close()

elif run_Backtest:
    plt.plot(sigma, IncomeBacktest, color='#2BBA58')
    plt.title(f"Strike {K2}, Time Horizon = {T} years, Fee = {(1-gamma)*100}%, RV = 2.56% annualized, Backtest USDC/USDT", fontsize=10)
    plt.xlabel("Implied Volatility Parameter", fontsize=10)
    plt.ylabel("Expected Fees", fontsize=10)
    plt.show()
    plt.close()