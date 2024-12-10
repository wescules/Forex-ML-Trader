from backtesting import Backtest, Strategy
import pandas as pd
import matplotlib.pyplot as plt
from backtesting.test import SMA, EURUSD, GOOG
import seaborn as sns
# Custom module import
import pkgutil
import inspect
from importlib import import_module

from strategies import *

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
strat = MeanRevision2


bt = Backtest(EURUSD, strat, commission=.002,
              cash=250000, exclusive_orders=True, trade_on_close=True)
# output = bt.run()
# print(output)


opt_values = strat.get_optimization_params()


stats, heatmap = bt.optimize(
    **opt_values,
    maximize='Equity Final [$]',
    max_tries=20000,
    method="grid",
    random_state=0,
    return_heatmap=True)
# heatmap = heatmap.sort_values().iloc[-10:]
# hm = heatmap.groupby(['rsi_window', 'length']).mean().unstack()
# sns.heatmap(hm[::-1], cmap='viridis')
# plt.show()

print(stats)


bt.plot()
