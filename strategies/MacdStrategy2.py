from backtesting import Strategy
import pandas as pd
import pandas_ta as ta
from backtesting.lib import crossover


class MacdStrategy2(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    S = 12
    L = 26
    H = 9

    current_state = 0   # 0 = None, 1 = Break UP, -1 = Break Down

    def init(self):
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), 14)

        # macd = diff between EMA12 and EMA26
        # signal = ema9 on macd
        # hist = diff between macd and signal
        self.macd, self.macd_signal, self.macd_hist = self.I(ta.macd,
                                                             pd.Series(
                                                                 self.data.Close),
                                                             MacdStrategy2.S,
                                                             MacdStrategy2.L,
                                                             MacdStrategy2.H)

    def next(self):

        stoploss_buy = 0.95 * self.data.Close[-1]
        stoploss_sell = 1.05 * self.data.Close[-1]
        takeprofit_buy = 1.10 * self.data.Close[-1]
        takeprofit_sell = 0.90 * self.data.Close[-1]

        cross = 0

        # BULLISH SIGNAL
        if crossover(self.macd, self.macd_signal):
            cross = 1
        # BEARISH SIGNAL
        elif crossover(self.macd_signal, self.macd):
            cross = -1

        if self.position.is_long and self.rsi[-1] >= 80:
            # print("Closing position because of RSI!")
            self.position.close()
            self.current_state == 0
            return

        if self.position.is_short and self.rsi[-1] <= 20:
            # print("Closing position because of RSI!")
            self.position.close()
            self.current_state == 0
            return

        if self.position and cross != 0:
            self.position.close()
            self.current_state == 0
            return

        if self.position and cross == 0:
            return

        if cross == 1:
            self.buy(sl=stoploss_buy, tp=takeprofit_buy)
        elif cross == -1:
            self.sell(sl=stoploss_sell, tp=takeprofit_sell)
