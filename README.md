# fee-generation-py

Calculates average fees earned over $N$ number of GBM price runs $P_t$, for a given RMM01 configuration $(K, \sigma, \tau)$. Allows us to see the rate of fee growth relative to volatility. Observed values based solely on required arbitrage volume.

I left the code relatively general so we can just add CFMMs and price processes as we see fit for testing. Functions in a simpy environment.

Next to add: Stable Volatility (Static RMM-01 pool) and OU Processes centered on 1

## Modules & Dependencies

Consists of only 3 modules: 

- ``Modules/CFMM.py`` contains CFMM logic
- ``Modules/Arbitrage.py`` contains Arbitrageur logic
- ``Modules/Sim.py`` contains simulation execution logic

Package dependencies include: ``numpy``, ``simpy``, ``scipy``, ``matplotlib.pyplot`` 
