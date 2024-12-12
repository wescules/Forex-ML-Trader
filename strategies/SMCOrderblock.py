from backtesting import Strategy
from numpy import nan
import pandas_ta as ta
import pandas as pd
from tradingpatterns.smc import smc

class SMCOrderblock(Strategy):
    lookback_period = 40
    z_score_threshold = 3

    @staticmethod
    def get_optimization_params():
        return dict(
            lookback_period=range(20, 50, 5),
            z_score_threshold=range(1, 3, 1),
        )
    def init(self):
        self.ohlcv = self.data.df.copy()
        self.ohlcv = self.ohlcv.iloc[:, :5]
        self.ohlcv.rename(columns={'time': 'timestamp', 'Open': 'open', 'High': 'high',
              'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        
        self.swing_highs_lows_data = smc.swing_highs_lows(self.ohlcv, swing_length=5)
        self.ob = smc.ob(self.ohlcv, self.swing_highs_lows_data)
        
        self.order_block = self.I(lambda: self.ob.OB)
        
        # window = 100
        # ob = pd.DataFrame()
        # for pos in range(window, len(self.ohlcv)):
        #     window_df = self.ohlcv.iloc[pos - window: pos]
        #     self.swing_highs_lows_data = smc.swing_highs_lows(window_df, swing_length=5)
        #     ob = pd.concat([ob, smc.ob(window_df, self.swing_highs_lows_data)], ignore_index=True)
        # self.order_block = self.I(lambda: ob.OB)
        
    def next(self):
        if self.order_block[-1] == 1:
            self.buy()
        elif self.order_block[-1] == -1:
            self.sell()