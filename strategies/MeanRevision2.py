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
    TP_OFFSET = 0.0007
    SL_OFFSET = 0.0017

    length = 135
    std = 2
    rsi_upper = 75
    rsi_lower = 35
    rsi_window = 14
    atr_len = 10

    @staticmethod
    def get_optimization_params():
        return dict(
            length=range(20, 200, 5),
            rsi_upper=range(55, 90, 5),
            rsi_lower=range(10, 45, 5),
            rsi_window=range(5, 50, 5),
            atr_len=range(5, 30, 5),
        )

    # optimal for aud_usd_h4 is (length=185,rsi_upper=70,rsi_lower=20,rsi_window=10,atr_len=5).
    def init(self):
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        self.rsi = self.I(ta.rsi, pd.Series(close), self.rsi_window)

        self.bbands = ta.bbands(
            pd.Series(close), length=self.length, stddev=self.std)
        lowerStr = 'BBL_' + str(self.length) + '_' + str(self.std) + '.0'
        upperStr = 'BBU_' + str(self.length) + '_' + str(self.std) + '.0'
        self.lower_bbands = self.I(lambda: self.bbands[lowerStr])
        self.upper_bbands = self.I(lambda: self.bbands[upperStr])

        self.atr = self.I(ta.atr, pd.Series(
            high), pd.Series(low), pd.Series(close), self.atr_len)

    def next(self):
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        lower_band = self.lower_bbands[-1]
        upper_band = self.upper_bbands[-1]
        atr_value = self.atr[-1]

        if price < lower_band and rsi < self.rsi_lower and not self.position.is_long:
            commission = price * 0.002
            tp = float(price + self.TP_OFFSET + commission)
            sl = float(price - self.SL_OFFSET - commission)
            self.buy()
        elif price > upper_band and rsi > self.rsi_upper and not self.position.is_short:
            commission = price * 0.002
            tp = float(price - self.TP_OFFSET - commission)
            sl = float(price + self.SL_OFFSET + commission)
            self.sell()
