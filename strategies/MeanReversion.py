from backtesting import Strategy
import pandas as pd
import pandas_ta as ta
from backtesting.test import SMA


class MeanReversion(Strategy):

    roll = 50
    position_size = 1

    def init(self):

        def std_3(arr, n):
            return pd.Series(arr).rolling(n).std() * 2

        self.he = self.data.Close
        self.he_mean = self.I(SMA, self.he, self.roll)
        self.he_std = self.I(std_3, self.he, self.roll)
        self.he_upper = self.he_mean + self.he_std
        self.he_lower = self.he_mean - self.he_std
        self.he_close = self.I(SMA, self.he, 1)

    def next(self):
        if self.he_close < self.he_lower:
            self.buy()

        if self.he_close > self.he_upper:
            self.sell()
