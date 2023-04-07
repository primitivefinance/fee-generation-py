import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run different Primitive CFMM simulation tests')

    parser.add_argument('--GBM', action='store_true', default=False, help='Run GBM RMM-01 simulation')
    parser.add_argument('--OU', action='store_true', default=False, help='Run OU Stable Volatility simulation')
    parser.add_argument('--Backtest', action='store_true', default=False, help='Run Backtest Stable Volatility simulation')
    parser.add_argument('--CS', type=int, default=1, help='Run Constant Sum Sanity Check')

    args = parser.parse_args()
    return args
