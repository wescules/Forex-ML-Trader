from backtesting import Strategy
import pandas as pd
import pandas_ta as ta
from backtesting.lib import barssince


class RsiOscillator(Strategy):
    # above a value sell, below buy
    upper_bound = 70
    lower_bound = 30
    rsi_window = 14
    position_size = 1
 
    def init(self):
        self.daily_rsi = self.I(ta.rsi, pd.Series(
            self.data.Close), self.rsi_window)

    def next(self):
        price = self.data.Close[-1]
        if self.daily_rsi[-1] > self.upper_bound and barssince(self.daily_rsi < self.upper_bound) == 3:
            self.sell(size=self.position_size)
            self.position.close()

        elif self.lower_bound > self.daily_rsi[-1]:
            self.buy(size=self.position_size)
