# fee-generation-py

Calculates average fees earned and standard deviation over $M$ number of random price path runs $P_t$ or backtests performance against USDC/USDT price history imported from the. Uniswap V3 subgraph, for a given RMM01 or Stable Portfolio pool configuration $(K, \sigma, \tau)$. Allows us to plot the rate of fee growth relative to the pools' volatility parameter. Observed values based solely on required arbitrage volume given a specified minimum arbitrage profit bound. Plots and exports pool volatility, fee growth, and variance data as csv after the simulation. Runs using a CLI.

I left the code relatively general so we can just add CFMMs and price processes as we see fit for testing. Functions using simpy environments in multithreaded execution.

## Modules & Dependencies

Consists of 6 modules: 

- ``Modules/CFMM.py`` contains CFMM logic
- ``Modules/Arbitrage.py`` contains Arbitrageur logic
- ``Modules/PriceGen.py`` contains price generation logic
- ``Modules/Sim.py`` contains simulation execution logic
- ``Modules/cli.py`` contains command-line interface bindings
- ``Modules/config.ini`` configuration file for simulation and pool parameters

Package dependencies include: ``numpy``, ``simpy``, ``scipy``, ``matplotlib.pyplot``, ``requests``, ``time``, ``concurrent.futures``, ``pandas``, ``configparser``, ``argparse``

## CLI Commands

The simulation runs using commands to specify which simulation process to run and plot. Each process runs $M$ simulation runs for each of the $G$ IV pool parameters and plots the average fee growth and standard deviation against the pool IV, then exports data as csv. The commands are as follows:

- ``python3 Sim.py --GBM`` runs a Geometric Brownian Motion process against an RMM-01 pool
- ``python3 Sim.py --OU`` runs an Ornstein-Uhlenbeck process centered at 1 against a Stable Portfolio pool
- ``python3 Sim.py --Backtest`` runs 6 months of USDC/USDT against a Stable Portfolio pool 
- ``python3 Sim.py --CS`` runs an Ornstein-Uhlenbeck process centered at 1 against a Constant Sum pool
