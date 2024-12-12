from backtesting import Strategy
import pandas as pd
import pandas_ta as ta


class CustomStrat(Strategy):
    TP_OFFSET = 0.0017
    SL_OFFSET = 0.0007

    length = 135
    std = 2
    rsi_upper = 75
    rsi_lower = 35
    rsi_window = 14
    atr_len = 10

    supertrend_length = 10
    supertrend_factor = 2

    adx_period = 7
    adx_threshold = 15
    
    @staticmethod
    def get_optimization_params():
        return dict(
            length=range(20, 200, 5),
            rsi_upper=range(55, 90, 5),
            rsi_lower=range(10, 45, 5),
            rsi_window=range(5, 50, 5),
            atr_len=range(5, 30, 5),
            adx_period=range(12, 30, 2),
            adx_threshold=range(10, 50, 5),
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

        self.st = ta.supertrend(pd.Series(self.data.High),
                                pd.Series(self.data.Low),
                                pd.Series(self.data.Close),
                                self.supertrend_length, self.supertrend_factor)
        lowerStr = 'SUPERTs_' + str(self.supertrend_length) + \
            '_' + str(self.supertrend_factor) + '.0'
        upperStr = 'SUPERTl_' + str(self.supertrend_length) + \
            '_' + str(self.supertrend_factor) + '.0'
        direction = 'SUPERTd_' + str(self.supertrend_length) + \
            '_' + str(self.supertrend_factor) + '.0'
        trend = 'SUPERT_' + str(self.supertrend_length) + '_' + \
            str(self.supertrend_factor) + '.0'
        self.lower_st = self.I(lambda: self.st[lowerStr])
        self.upper_st = self.I(lambda: self.st[upperStr])
        self.direction = self.I(lambda: self.st[direction])
        self.trend = self.I(lambda: self.st[trend])
        
        self.adx = self.I(ta.adx, pd.Series(high), pd.Series(low), pd.Series(close), self.adx_period)

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
            self.buy(size=10000, tp=tp, sl=sl)
        elif price > upper_band and rsi > self.rsi_upper and not self.position.is_short:
            commission = price * 0.002
            tp = float(price - self.TP_OFFSET - commission)
            sl = float(price + self.SL_OFFSET + commission)
            self.sell(size=10000, tp=tp, sl=sl)
