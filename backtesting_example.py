from backtesting import Backtest
import pandas as pd
import matplotlib.pyplot as plt
from backtesting.test import EURUSD, GOOG
from backtesting.lib import random_ohlc_data
import seaborn as sns

from strategies import *

# CONSTANTS
KEY_INDICATORS = ["Return (Ann.) [%]", "Exposure Time [%]", "Volatility (Ann.) [%]",
                  "Return [%]", "Sharpe Ratio", "Buy & Hold Return [%]"]

STRATEGIES_DIR = 'strategies'

WINDOWS = ['M5', 'M15', 'M30', 'H1', 'H4', 'D']

class BacktestRunner():

    def __init__(self, price_data, strategy, margin: float = 1, commission: float = 0.002, cash: float = 10000,
                 exclusive_orders: bool = True, trade_on_close: bool = True, filename: str = None):
        self.bt = Backtest(data=price_data, strategy=strategy, commission=commission, margin=margin,
                           cash=cash, exclusive_orders=exclusive_orders, trade_on_close=trade_on_close)
        self.strategy = strategy
        self.filename = filename

    def run(self):
        stats = self.bt.run()
        print(stats)
        self.bt.plot(filename=self.filename)

    def run_optimized(self, show_heatmap: bool = False):
        opt_values = self.strategy.get_optimization_params()

        stats, heatmap = self.bt.optimize(
            **opt_values,
            maximize='Equity Final [$]',
            max_tries=20000,
            method="grid",
            random_state=0,
            return_heatmap=True)
        print(stats)
        self.bt.plot(filename=self.filename)
        
        # show a visual display of what the the params bring in the $$$
        if show_heatmap:
            hm = heatmap.groupby(list(opt_values.keys())).mean().unstack()
            # print(hm)
            sns.heatmap(hm[::-1], cmap='viridis')
            plt.show()


def run_multiple_timeframes(currency_pair, timeframes):
    for timeframe in timeframes:
        filename = 'history/' + currency_pair + '_' + timeframe + '.csv'
        price_data = pd.read_csv(filename,
                                 index_col=0, header=0, parse_dates=True)
        price_data.rename(columns={'time': ''}, inplace=True)
        price_data = price_data.iloc[:, :5]
        
        plotting_filename = '_' + currency_pair + '_' + timeframe

        backtest = BacktestRunner(price_data, SMCOrderblock, commission=.002, margin=0.002,
                                  cash=5000, exclusive_orders=False, trade_on_close=True, filename=plotting_filename)

        backtest.run()


if __name__ == "__main__":
    # run_multiple_timeframes('EUR_USD', WINDOWS)
    run_multiple_timeframes('EUR_USD', ['M30', 'H1', "H4"])
