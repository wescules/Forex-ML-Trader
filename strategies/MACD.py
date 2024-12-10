from backtesting import Strategy
import pandas as pd
import pandas_ta as ta
from backtesting.lib import barssince, crossover
import numpy as np


class MACD(Strategy):
    # when macd crosses 0 from above, bear indication & vice-versa
    macd_fast = 12
    macd_slow = 26
    position_size = 50
    tp_over_macd = 10
    tp_under_macd = 10
    
    @staticmethod
    def get_optimization_params():
        return dict(
            macd_slow=range(12, 50, 2),
            macd_fast=range(4, 30, 2),
        )

    def init(self):
        def zero_line(arr):
            return np.full_like(arr, 0)

        self.macd_close = self.data.Close
        self.macd_values, self.macd_signal, self.macd_hist = self.I(
            ta.macd, pd.Series(self.data.Close), self.macd_fast, self.macd_slow, 20)
        self.zero = self.I(zero_line, self.macd_close)

    def next(self):
        price = self.data.Close

        # macd crossing zero from above, short stock
        if crossover(self.zero, self.macd_values):
            self.position.close()
            self.sell()

        # if a short exists and macd stays below zero for 5 consecutive days, close short position
        if self.position.is_short and barssince(self.macd_values > 0) == self.tp_over_macd:
            self.position.close()
        elif self.position.is_long and barssince(self.macd_values < 0) == self.tp_under_macd:
            self.position.close()

        # crossing from below zero line, buy stock
        if crossover(self.macd_values, self.zero):
            self.buy()
