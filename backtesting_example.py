from backtesting import Backtest
import pandas as pd
import matplotlib.pyplot as plt
from backtesting.test import EURUSD, GOOG
import seaborn as sns

from strategies import *

# CONSTANTS
KEY_INDICATORS = ["Return (Ann.) [%]", "Exposure Time [%]", "Volatility (Ann.) [%]",
                  "Return [%]", "Sharpe Ratio", "Buy & Hold Return [%]"]

STRATEGIES_DIR = 'strategies'


class BacktestRunner():

    def __init__(self, price_data, strategy, margin: float = 1, commission: float = 0.002, cash: float = 10000,
                 exclusive_orders: bool = True, trade_on_close: bool = True):
        self.bt = Backtest(data=price_data, strategy=strategy, commission=commission, margin=margin,
                           cash=cash, exclusive_orders=exclusive_orders, trade_on_close=trade_on_close)
        self.strategy = strategy

    def run(self):
        stats = self.bt.run()
        print(stats)
        self.bt.plot()

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
        self.bt.plot()
        
        # show a visual display of what the the params bring in the $$$
        if show_heatmap:
            hm = heatmap.groupby(list(opt_values.keys())).mean().unstack()
            # print(hm)
            sns.heatmap(hm[::-1], cmap='viridis')
            plt.show()



if __name__ == "__main__":
    price_data = pd.read_csv('history/AUD_USD_H1.csv',
                             index_col=0, header=0, parse_dates=True)
    price_data.rename(columns={'time': ''}, inplace=True)
    price_data = price_data.iloc[:, :5]

    backtest = BacktestRunner(price_data, DoubleSuperTrend, commission=.002, margin=1,
                              cash=5000, exclusive_orders=True, trade_on_close=True)

    backtest.run_optimized()
