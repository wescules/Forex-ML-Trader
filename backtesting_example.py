from backtesting import Backtest, Strategy
import pandas as pd

from backtesting.test import SMA, EURUSD, GOOG


# Custom module import
import os
import pkgutil
import inspect
from importlib import import_module

from strategies import MACD


# CONSTANTS
KEY_INDICATORS = ["Return (Ann.) [%]", "Exposure Time [%]", "Volatility (Ann.) [%]",
                  "Return [%]", "Sharpe Ratio", "Buy & Hold Return [%]"]

STRATEGIES_DIR = 'strategies'

def get_strategy(strat: str):
    return get_all_strategies(strategy_name=strat)

def get_all_strategies(strategy_name: str = None):
    strategy_classes = {}
    for _, module_name, _ in pkgutil.iter_modules([STRATEGIES_DIR]):
        # Import the strategy module dynamically
        module = import_module(f'{STRATEGIES_DIR}.{module_name}')

        # Get the strategy class from the module
        strategy_class = next((cls for _, cls in inspect.getmembers(
            module, inspect.isclass) if issubclass(cls, Strategy)), None)

        # Add the strategy class to the dictionary with the module name as key
        if strategy_class is not None:
            strategy_classes[module_name] = strategy_class
            
        if strategy_name and module_name == strategy_name:
            return strategy_class
            
    return strategy_classes


price_data = pd.read_csv('history/AUD_USD_H4.csv',
                         index_col=0, header=0, parse_dates=True)
price_data.rename(columns={'time': ''}, inplace=True)

price_data = price_data.iloc[:, :5]
# strat = get_strategy('BbandCross')
strat = get_strategy('MeanRevision2')


bt = Backtest(price_data, strat, commission=.002,
              cash=5000, exclusive_orders=True)
output = bt.run()
print(output)


# stats, heatmap = bt.optimize(
#     n1=range(10, 110, 10),
#     n2=range(20, 210, 20),
#     n_enter=range(15, 35, 5),
#     n_exit=range(10, 25, 5),
#     constraint=lambda p: p.n_exit < p.n_enter < p.n1 < p.n2,
#     maximize='Equity Final [$]',
#     max_tries=200,
#     method="grid",
#     random_state=0,
#     return_heatmap=True)

# stats, heatmap = bt.optimize(
#     S=range(1, 36, 1),
#     L=range(1, 36, 1),
#     H=range(1, 30, 1),
#     maximize='Equity Final [$]',
#     max_tries=200,
#     method="grid",
#     random_state=0,
#     return_heatmap=True)
# heatmap = heatmap.sort_values().iloc[-3:]
# print(heatmap)


bt.plot()
