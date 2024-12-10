from backtesting import Strategy
import pandas as pd
import pandas_ta as ta
from backtesting.lib import crossover

class SuperTrend(Strategy):
    # above a value sell, below buy
    length = 10
    supertrend_factor = 2

    @staticmethod
    def get_optimization_params():
        return dict(
            length=range(10, 30, 2),
            supertrend_factor=range(1, 3, 1),
        )

    def init(self):
        
        self.st = ta.supertrend(pd.Series(self.data.High),
                                pd.Series(self.data.Low),
                                pd.Series(self.data.Close),
                                self.length, self.supertrend_factor)
        lowerStr = 'SUPERTs_' + str(self.length) + '_' + str(self.supertrend_factor) + '.0'
        upperStr = 'SUPERTl_' + str(self.length) + '_' + str(self.supertrend_factor) + '.0'
        direction = 'SUPERTd_' + str(self.length) + '_' + str(self.supertrend_factor) + '.0'
        trend = 'SUPERT_' + str(self.length) + '_' + str(self.supertrend_factor) + '.0'
        self.lower_st = self.I(lambda: self.st[lowerStr])
        self.upper_st = self.I(lambda: self.st[upperStr])
        self.direction = self.I(lambda: self.st[direction])
        self.trend = self.I(lambda: self.st[trend])

    def next(self):
        if self.direction[-1] == 1 and not self.position.is_long:
            self.buy()
        elif self.direction[-1] == -1 and not self.position.is_short:
            self.sell()
