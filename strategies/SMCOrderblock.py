from backtesting import Strategy
from tradingpatterns.smc import smc


class SMCOrderblock(Strategy):
    swing_length = 4
    evaluator = smc.SwingMethodEvaluator.MOMENTUM
    long_swing_len = 15

    @staticmethod
    def get_optimization_params():
        return dict(
            swing_length=range(1, 15, 1),
            long_swing_len=range(15, 30, 5),
            evaluator=[smc.SwingMethodEvaluator.COMBINED, smc.SwingMethodEvaluator.MOMENTUM, smc.SwingMethodEvaluator.DEFAULT,
                       smc.SwingMethodEvaluator.FRACTALS, smc.SwingMethodEvaluator.WEIGHTED_ROLLING_WINDOW]
        )

    def init(self):
        self.ohlcv = self.data.df.copy()
        self.ohlcv = self.ohlcv.iloc[:, :5]
        self.ohlcv.rename(columns={'time': 'date', 'Open': 'open', 'High': 'high',
                                   'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

        self.swing_highs_lows_data = smc.swing_highs_lows(self.ohlcv, swing_length=self.swing_length, swing_evaluator=self.evaluator,
                                                          short_swing_length=self.swing_length, long_swing_length=self.long_swing_len)
        self.ob = smc.ob(self.ohlcv, self.swing_highs_lows_data)
        self.order_block = self.I(lambda: self.ob.OB)

    def next(self):
        if self.order_block[-1] == 1:
            tp = self.data.Close[-1] + 0.007
            self.buy(size=10000, tp=tp)
        elif self.order_block[-1] == -1:
            tp = self.data.Close[-1] - 0.007
            self.sell(size=10000, tp=tp)
