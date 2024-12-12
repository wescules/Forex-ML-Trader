from backtesting import Strategy
import pandas as pd
import pandas_ta as ta

class DoubleSuperTrend(Strategy):
    # the longer the length the more candles you need until it starts trading
    short_supertrend_length = 10
    short_supertrend_factor = 3
    long_supertrend_length = 85
    long_supertrend_factor = 8
    
    adx_period = 7
    
    upper_bound = 70
    lower_bound = 30
    rsi_window = 14

    length = 20
    std = 2

    @staticmethod
    def get_optimization_params():
        return dict(
            short_supertrend_length=range(5, 100, 5),
            short_supertrend_factor=range(1, 5, 1),
            long_supertrend_length=range(30, 100, 5),
            long_supertrend_factor=range(2, 10, 1),
            adx_period=range(12, 30, 2),
            constraint=lambda p: p.short_supertrend_length < p.long_supertrend_length,
        )

    def init(self):
        self.short_st = ta.supertrend(pd.Series(self.data.High),
                                      pd.Series(self.data.Low),
                                      pd.Series(self.data.Close),
                                      self.short_supertrend_length, self.short_supertrend_factor)
        lowerStr = 'SUPERTs_' + str(self.short_supertrend_length) + \
            '_' + str(self.short_supertrend_factor) + '.0'
        upperStr = 'SUPERTl_' + str(self.short_supertrend_length) + \
            '_' + str(self.short_supertrend_factor) + '.0'
        direction = 'SUPERTd_' + str(self.short_supertrend_length) + \
            '_' + str(self.short_supertrend_factor) + '.0'
        trend = 'SUPERT_' + str(self.short_supertrend_length) + '_' + \
            str(self.short_supertrend_factor) + '.0'
        self.short_lower_st = self.I(lambda: self.short_st[lowerStr])
        self.short_upper_st = self.I(lambda: self.short_st[upperStr])
        self.short_direction = self.I(lambda: self.short_st[direction])
        self.short_trend = self.I(lambda: self.short_st[trend])

        # long super trend
        self.long_st = ta.supertrend(pd.Series(self.data.High),
                                     pd.Series(self.data.Low),
                                     pd.Series(self.data.Close),
                                     self.long_supertrend_length, self.long_supertrend_factor)
        lowerStr = 'SUPERTs_' + str(self.long_supertrend_length) + \
            '_' + str(self.long_supertrend_factor) + '.0'
        upperStr = 'SUPERTl_' + str(self.long_supertrend_length) + \
            '_' + str(self.long_supertrend_factor) + '.0'
        direction = 'SUPERTd_' + str(self.long_supertrend_length) + \
            '_' + str(self.long_supertrend_factor) + '.0'
        trend = 'SUPERT_' + str(self.long_supertrend_length) + '_' + \
            str(self.long_supertrend_factor) + '.0'
        self.long_lower_st = self.I(lambda: self.long_st[lowerStr])
        self.long_upper_st = self.I(lambda: self.long_st[upperStr])
        self.long_direction = self.I(lambda: self.long_st[direction])
        self.long_trend = self.I(lambda: self.long_st[trend])


        high, low, close = pd.Series(self.data.High), pd.Series(
            self.data.Low), pd.Series(self.data.Close)
        self.adx = self.I(ta.adx, high, low, close, self.adx_period)

        
        self.daily_rsi = self.I(ta.rsi, pd.Series(
            self.data.Close), self.rsi_window)

        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        self.bbands = ta.bbands(pd.Series(close), length=self.length, stddev=self.std)
        lowerStr = 'BBL_' + str(self.length) + '_' + str(self.std) + '.0'
        upperStr = 'BBU_' + str(self.length) + '_' + str(self.std) + '.0'
        self.lower_bbands = self.I(lambda: self.bbands[lowerStr])
        self.upper_bbands = self.I(lambda: self.bbands[upperStr])
    def next(self):
        for trade in self.trades:
            if trade.is_long and self.short_direction == -1:
                trade.close()
            elif trade.is_short and self.short_direction == 1:
                trade.close()
                
        if self.adx > 25 and self.long_direction[-1] == 1 and self.short_direction == self.long_direction and not self.position.is_long:
            self.buy()
        elif self.adx > 25 and self.long_direction[-1] == -1 and self.short_direction == self.long_direction and not self.position.is_short:
            self.sell()
