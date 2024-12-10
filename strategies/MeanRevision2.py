from backtesting import Strategy
import pandas as pd
import pandas_ta as ta


class MeanRevision2(Strategy):
    # 2. Mean Reversion Strategy
    # Idea: Assume price will revert to the mean after significant moves.

    # Indicators: Bollinger Bands, RSI.
    # Entry/Exit Rules:
    #     Go long when the price closes below the lower Bollinger Band and RSI < 30.
    #     Go short when the price closes above the upper Bollinger Band and RSI > 70.
    #     Exit when the price returns to the middle Bollinger Band.
    # Stop-Loss/Take-Profit:
    #     Place stop-loss slightly outside the band on entry.
    #     Take profit at the Bollinger midline or a predefined risk-reward ratio (e.g., 2:1).

    length = 20
    std = 2
    rsi_upper = 70
    rsi_lower = 30
    rsi_window = 14

    def init(self):
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), self.rsi_window)
        
        self.bbands = ta.bbands(pd.Series(self.data.Close), length=self.length, stddev=self.std)
        lowerStr = 'BBL_' + str(self.length) + '_' + str(self.std) + '.0'
        upperStr = 'BBU_' + str(self.length) + '_' + str(self.std) + '.0'        
        self.lower_bbands = self.I(lambda: self.bbands[lowerStr])
        self.upper_bbands = self.I(lambda: self.bbands[upperStr])

    def next(self):
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        lower_band = self.lower_bbands[-1]
        upper_band = self.upper_bbands[-1]

        if price < lower_band and rsi < self.rsi_lower:
            self.buy()
        elif price > upper_band and rsi > self.rsi_upper:
            self.sell()
