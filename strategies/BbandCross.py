from backtesting import Backtest
from backtesting.lib import crossover, TrailingStrategy
import pandas_ta as ta
import pandas as pd


class BbandCross(TrailingStrategy):
    length = 20
    std = 2
    atr_len = 14

    @staticmethod
    def get_optimization_params():
        return dict(
            length=range(20, 200, 5),
            atr_len=range(5, 30, 5)
        )

    def init(self):
        super().init()
        self.set_trailing_sl(2.5)

        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        self.bbands = ta.bbands(pd.Series(close), length=self.length, stddev=self.std)
        lowerStr = 'BBL_' + str(self.length) + '_' + str(self.std) + '.0'
        upperStr = 'BBU_' + str(self.length) + '_' + str(self.std) + '.0'
        self.lower_bbands = self.I(lambda: self.bbands[lowerStr])
        self.upper_bbands = self.I(lambda: self.bbands[upperStr])

        self.atr = self.I(ta.atr, pd.Series(
            high), pd.Series(low), pd.Series(close), self.atr_len)

    def next(self):
        super().next()

        if crossover(self.data.Close, self.upper_bbands[-1]):
            self.sl = self.data.Close[-1] - 2.5*self.atr[-1]
            self.buy(size=0.75, sl=self.sl)

        elif crossover(self.lower_bbands[-1], self.data.Close):
            self.sl = self.data.Close[-1] + 2.5*self.atr[-1]
            self.sell(size=0.75, sl=self.sl)
