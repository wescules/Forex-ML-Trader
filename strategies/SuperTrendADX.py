from backtesting import Strategy
import pandas as pd
import pandas_ta as ta


class SuperTrendADX(Strategy):
    # the longer the length the more candles you need until it starts trading
    length = 100
    supertrend_factor = 3
    adx_period = 7

    @staticmethod
    def get_optimization_params():
        return dict(
            length=range(5, 600, 10),
            supertrend_factor=range(1, 5, 1),
            adx_period=range(12, 30, 2),
        )

    def init(self):
        self.st = ta.supertrend(pd.Series(self.data.High),
                                pd.Series(self.data.Low),
                                pd.Series(self.data.Close),
                                self.length, self.supertrend_factor)
        lowerStr = 'SUPERTs_' + str(self.length) + \
            '_' + str(self.supertrend_factor) + '.0'
        upperStr = 'SUPERTl_' + str(self.length) + \
            '_' + str(self.supertrend_factor) + '.0'
        direction = 'SUPERTd_' + str(self.length) + \
            '_' + str(self.supertrend_factor) + '.0'
        trend = 'SUPERT_' + str(self.length) + '_' + \
            str(self.supertrend_factor) + '.0'
        self.lower_st = self.I(lambda: self.st[lowerStr])
        self.upper_st = self.I(lambda: self.st[upperStr])
        self.direction = self.I(lambda: self.st[direction])
        self.trend = self.I(lambda: self.st[trend])

        high, low, close = pd.Series(self.data.High), pd.Series(
            self.data.Low), pd.Series(self.data.Close)
        self.adx = self.I(ta.adx, high, low, close, self.adx_period)

    def next(self):

        if self.direction[-1] == 1 and self.data.Close[-1] > self.upper_st[-1] and not self.position.is_long:
            self.buy()
        elif self.direction[-1] == -1 and self.data.Close[-1] < self.lower_st[-1] and self.adx > 50 and not self.position.is_short:
            self.sell()
