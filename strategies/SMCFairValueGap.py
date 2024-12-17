from backtesting import Strategy
from tradingpatterns.smc import smc


# this strat sucks.. very low returns and produces too many signals.

class SMCFairValueGap(Strategy):
    swing_length = 4

    @staticmethod
    def get_optimization_params():
        return dict(
            swing_length=range(1, 15, 1),)

    def init(self):
        self.ohlcv = self.data.df.copy()
        self.ohlcv = self.ohlcv.iloc[:, :5]
        self.ohlcv.rename(columns={'time': 'date', 'Open': 'open', 'High': 'high',
                                   'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

        self.df = smc.fvg(self.ohlcv)
        self.fvg = self.I(lambda: self.df.FVG)

    def next(self):
        if self.fvg[-1] == 1 and not self.position.is_long:
            tp = self.data.Close[-1] + 0.007
            self.buy(size=10000)
        elif self.fvg[-1] == -1 and not self.position.is_long:
            tp = self.data.Close[-1] - 0.007
            self.sell(size=10000)
