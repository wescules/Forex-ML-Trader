from backtesting import Strategy
import pandas_ta as ta
import pandas as pd
class MeanReversionBollinger(Strategy):
    lookback_period = 40
    z_score_threshold = 3

    @staticmethod
    def get_optimization_params():
        return dict(
            lookback_period=range(20, 50, 5),
            z_score_threshold=range(1, 3, 1),
        )
    def init(self):
        close = pd.Series(self.data.Close)
        self.mean = self.I(ta.sma, close, self.lookback_period)
        self.std = self.I(ta.stdev, close, self.lookback_period)
        self.upper_band = self.mean + self.z_score_threshold * self.std
        self.lower_band = self.mean - self.z_score_threshold * self.std

    def next(self):
        if self.data.Close[-1] > self.upper_band[-1] and not self.position.is_short:
            self.sell()
        elif self.data.Close[-1] < self.lower_band[-1] and not self.position.is_long:
            self.buy()