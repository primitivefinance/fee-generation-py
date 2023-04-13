import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import simpy
import configparser
import pandas as pd
import os

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

# Simulation Controls

run_GBM_simulation = args.GBM
run_OU_simulation = args.OU
run_Backtest = args.Backtest
run_ConstantSum_test = args.CS
run_Backtest_OptimizedC = args.OptimizedTest

# config.ini Parameters

G = int(config["Sim"]["G"])
sigma_low = float(config["Sim"]["sigma_low"])
sigma_high = float(config["Sim"]["sigma_high"])
K1 = float(config["Pool"]["strike_RMM01"])
p0 = float(config["Pool"]["P0_RMM01"])
K2 = float(config["Pool"]["strike_SV"])
P0 = float(config["Pool"]["P0_SV"])
sigma = np.linspace(sigma_low, sigma_high, G)
T = float(config["Pool"]["time_horizon"])*0.0027397260273972603
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

# RMM-01 arbitraged against an infinitely liquid Geometric Brownian Motion (GBM) reference market

def simulateGBM(env, i, Fees):
    '''
    Performs arbitrage between RMM-01 and a GBM price process at each simulation timestep and records the arbitrage fees.
    '''
    Curve = CFMM.RMM01(p0, K1, sigma[i], T, dt, gamma, env, shares)
    
    while True:

        GBM = price.generateGBM(T, mu, v, p0, dt, env)
        arb = a(GBM, Curve, Arb)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

def GBMSimProcess(j):
    '''
    Runs M simulations of the simulateGBM process and returns the average and variance of the arbitrage fees earned.
    '''
    FeeIncome = []
    for i in range (0, M):
        Fees = []
        env = simpy.Environment()
        env.process(simulateGBM(env, j, Fees))
        env.run(until=N)

        FeeIncome.append(sum(Fees))
    avgIncome = sum(FeeIncome)/M
    varIncome = sum([(fee - avgIncome)**2 for fee in FeeIncome])/M
    print(f"{j/G*100} Complete")
    return avgIncome, varIncome

## Multithreaded execution against different implied volatility parameters

avgIncomeGBM = []
varIncomeGBM = []
if run_GBM_simulation:
    array = []
    with ThreadPoolExecutor() as executor:
        array = list(executor.map(GBMSimProcess, range(G)))
    avgIncomeGBM, varIncomeGBM = zip(*array)

stdIncomeGBM = [np.sqrt(var) for var in varIncomeGBM]

# OU Based Stable Volatility simulation

def simulateOU(env, i, Fees):
    '''
    Performs arbitrage between Stable Volatility and a Ornstein-Uhlenbeck price process at each simulation timestep and records the arbitrage fees.
    '''
    Curve = CFMM.StableVolatility(P0, K2, sigma[i], T, gamma, env, shares)

    while True:

        OU = price.generateOU(T, mean, v, P0, dt, theta, env)
        arb = a(OU, Curve, Arb)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

def OUSimProcess(j):
    '''
    Runs M simulations of the simulateOU process and returns the average and variance of the arbitrage fees earned.
    '''
    FeeIncome = []
    for i in range (0, M):
        Fees = []
        env = simpy.Environment()
        env.process(simulateOU(env, j, Fees))
        env.run(until=N)

        FeeIncome.append(sum(Fees))
    avgIncome = sum(FeeIncome)/M
    varIncome = sum([(fee - avgIncome)**2 for fee in FeeIncome])/M
    print(f"{j/G*100} Complete")
    return avgIncome, varIncome

## Multithreaded execution against different implied volatility parameters

avgIncomeOU = []
varIncomeOU = []
if run_OU_simulation:
    array = []
    with ThreadPoolExecutor() as executor:
        array = list(executor.map(OUSimProcess, range(G)))
    avgIncomeOU, varIncomeOU = zip(*array)

stdIncomeOU = [np.sqrt(var) for var in varIncomeOU]

# Backtest Stable Volatility simulation

def simulateBacktest(env, i, Fees):
    '''
    Performs arbitrage at each simulation timestep between Stable Volatility and USDC/USDT price data from the Uniswap V3 subgraph and records the arbitrage fees.
    '''
    Curve = CFMM.StableVolatility(P0, K2, sigma[i], T, gamma, env, shares)

    while True:

        arb = a(price.close_values[env.now], Curve, Arb)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

def BacktestProcess(j):
    '''
    Runs a single simulation of the simulateBacktest process and returns the arbitrage fees earned.
    '''
    FeeIncome = []
    Fees = []
    env = simpy.Environment()
    env.process(simulateBacktest(env, j, Fees))
    env.run(until=len(price.close_values))

    FeeIncome.append(sum(Fees))
    print(f"{j/G*100} Complete")
    return sum(FeeIncome)

## Multithreaded execution against different implied volatility parameters

IncomeBacktest = []
if run_Backtest:
    with ThreadPoolExecutor() as executor:
        IncomeBacktest = list(executor.map(BacktestProcess, range(G)))

# Constant Sum OU Test

def simulateConstantSum(env):
    Curve = CFMM.ConstantSum(K2, shares/2, shares/2, gamma, env)

    while True:
        OU = price.generateOU(T, mean, v, P0, dt, theta, env)
        arb = a(OU, Curve, Arb)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

if run_ConstantSum_test:
    FeeIncome = []
    for i in range (0, M):
        Fees = []
        env = simpy.Environment()
        env.process(simulateConstantSum(env))
        env.run(until=N)
        FeeIncome.append(sum(Fees))
    FeeIncome = sum(FeeIncome)/M
    print(FeeIncome)

# 0.000415 c value Backtest

def simulateBacktest_OptimizedC(env, Fees):
    '''
    Performs arbitrage at each simulation timestep between Stable Volatility and USDC/USDT price data from the Uniswap V3 subgraph and records the arbitrage fees.
    T must be 7/365 for this to work.
    '''
    Curve = CFMM.StableVolatility(P0, K2, 0.0005, T, gamma, env, shares)

    while True:

        arb = a(price.close_values[env.now], Curve, Arb)
        arb.arbitrage()
        Fees.append(arb.Fees)
        yield env.timeout(1)

if run_Backtest_OptimizedC:
    Fees = []
    env = simpy.Environment()
    env.process(simulateBacktest_OptimizedC(env, Fees))
    env.run(until=len(price.close_values))
    FeeIncome = sum(Fees)
    print("Fees Earned:", FeeIncome)

# Plotting and Data Export    

if run_GBM_simulation:
    '''
    Plotting implied volatility parameter vs. average fees generated for GBM & RMM-01 along with error bars.
    Exports data to CSV
    '''
    plt.plot(sigma, avgIncomeGBM, color='#2BBA58')
    plt.errorbar(sigma, avgIncomeGBM, yerr=stdIncomeGBM, color='#2BBA58', ecolor='#BCF2CD', capthick=2)
    plt.title(f"Strike {K1}, Time Horizon = {T} years, Fee = {(1-gamma)*100}%, RV = {v*100}% annualized, Drift = {mu*100}%, GBM RMM-01 Simulation", fontsize=10) 
    plt.xlabel("Implied Volatility Parameter", fontsize=10)
    plt.ylabel("Expected Fees", fontsize=10)
    plt.show()
    plt.close()

    data = {'Column1': sigma, 'Column2': avgIncomeGBM, 'Column3': stdIncomeGBM}
    df = pd.DataFrame(data)

    output_directory = "csv-data/GBM"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file_name = f"GBMTest{M}Runs{G}IVs{sigma_low}to{sigma_high}.csv"

    df.to_csv(os.path.join(output_directory, output_file_name), index=False)

elif run_OU_simulation:
    '''
    Plotting implied volatility parameter vs. average fees generated for OU & Stable Volatility along with error bars.
    Exports data to CSV
    '''
    plt.plot(sigma, avgIncomeOU, color='#2BBA58')
    plt.errorbar(sigma, avgIncomeOU, yerr=stdIncomeOU, color='#2BBA58', ecolor='#BCF2CD', capthick=2)
    plt.title(f"Strike {K2}, Time Horizon = {T} years, Fee = {(1-gamma)*100}%, RV = {v*100}% annualized, Mean Price = {mean}, Theta = {theta}, OU Stable Volatility Simulation", fontsize=10)
    plt.xlabel("Implied Volatility Parameter", fontsize=10)
    plt.ylabel("Expected Fees", fontsize=10)
    plt.legend()
    plt.show()
    plt.close()

    data = {'Column1': sigma, 'Column2': avgIncomeOU, 'Column3': stdIncomeOU}
    df = pd.DataFrame(data)

    output_directory = "csv-data/OU"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file_name = f"OUTest{M}Runs{G}IVs{sigma_low}to{sigma_high}.csv"

    df.to_csv(os.path.join(output_directory, output_file_name), index=False)

elif run_Backtest:
    '''
    Plotting implied volatility parameter vs. average fees generated for USDC/USDT Backtest Data & Stable Volatility.
    Exports data as CSV.
    '''
    plt.plot(sigma, IncomeBacktest, color='#2BBA58')
    plt.title(f"Strike {K2}, Time Horizon = {T} years, Fee = {(1-gamma)*100}%, RV = 2.56% annualized, Backtest USDC/USDT", fontsize=10)
    plt.xlabel("Implied Volatility Parameter", fontsize=10)
    plt.ylabel("Expected Fees", fontsize=10)
    plt.show()
    plt.close()

    data = {'Column1': sigma, 'Column2': IncomeBacktest}
    df = pd.DataFrame(data)

    output_directory = "csv-data/Backtest"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file_name = f"USDCUSDT_Backtest{G}IVs{sigma_low}to{sigma_high}.csv"

    df.to_csv(os.path.join(output_directory, output_file_name), index=False)