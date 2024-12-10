from backtesting import Strategy
import pandas as pd
import pandas_ta as ta


class AverageDirectionalMovement(Strategy):
    # ADX Period:
    # The ADX period determines the number of bars used to calculate the ADX indicator. A shorter period makes
    # the strategy more responsive to recent price changes, while a longer period provides a smoother ADX line.
    adx_period = 7

    # ADX Threshold:
    # The ADX threshold is the level above which the ADX value must cross to generate a buy signal. It determines
    # the strength of the trend required to trigger a trade. A higher threshold filters out weaker trends and
    # reduces the frequency of trades, while a lower threshold allows more trades but may result in entering
    # weaker trends.
    adx_threshold = 15
    
    @staticmethod
    def get_optimization_params():
        return dict(
            adx_period=range(12, 30, 2),
            adx_threshold=range(10, 50, 5),
        )


    def init(self):
        high, low, close = self.data.High, self.data.Low, self.data.Close
        # Calculate Average Directional Index (ADX) using TA-Lib
        self.adx = self.I(ta.adx, high, low, pd.Series(close), self.adx_period)

    def next(self):
        # Check if ADX value is above the threshold
        if self.adx[-1] > self.adx_threshold:
            # Check if the previous ADX value was below the threshold
            if self.adx[-2] < self.adx_threshold:
                # Generate a buy signal when ADX crosses above the threshold
                self.buy()
        else:
            # Generate a sell signal when ADX falls below the threshold
            self.sell()
