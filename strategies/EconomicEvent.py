from backtesting import Strategy
import pandas as pd
import pandas_ta as ta
from backtesting.lib import barssince, crossover
import numpy as np


econ_calendar = pd.read_csv("forexscraper/economic_calendar.csv", parse_dates=["datetime"])

# Preprocess calendar to include only high-impact events
econ_calendar = econ_calendar[econ_calendar["impact"] == "High"]

class EconomicEvent(Strategy):
    # when macd crosses 0 from above, bear indication & vice-versa
    atr_len = 14
    
    @staticmethod
    def get_optimization_params():
        return dict(
            atr_len=range(5, 30, 5)
        )
        
    def calculate_position_size(self, entry_price, sl_price):
        # Account balance and risk amount
        account_balance = self.equity *50
        risk_amount = account_balance * self.risk_percent

        # Calculate position size based on risk
        risk_per_trade = abs(entry_price - sl_price)
        position_size = risk_amount / risk_per_trade if risk_per_trade > 0 else 0
        return position_size

    risk_percent = 0.01  # Risk 1% of account balance
    cooldown_period = 5  # Number of candles to wait before a new trade
    atr_multiplier = 2.0  # ATR multiplier for dynamic SL/TP

    def init(self):
        # Merge economic calendar with price data
        self.econ_events = econ_calendar.set_index("datetime")

        # Calculate ATR for dynamic SL/TP
        self.atr = self.atr = self.I(ta.atr, pd.Series(
            self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), self.atr_len)

        # Store the last trade's time index
        self.last_trade_time = -self.cooldown_period


    def next(self):
        current_time = self.data.index[-1]

        # Check for an economic event at the current time
        if current_time in self.econ_events.index:
            event = self.econ_events.loc[current_time]

            # Skip if within cooldown period
            # if len(self.trades) > 0 and self.trades[-1].exit_bar > (len(self.data) - self.cooldown_period):
            #     return

            # Risk management: calculate position size based on ATR
            atr_value = self.atr[-1]
            if np.isnan(atr_value):
                return  # Skip if ATR is not available yet
            self.data
            entry_price = self.data.Close[-1]
            sl_price = entry_price - self.atr_multiplier * atr_value
            tp_price = entry_price + self.atr_multiplier * atr_value
            position_size = self.calculate_position_size(entry_price, sl_price)
            
            # Example logic: Go long on USD-related events
            currency1, currency2 = self.currency_pair
            if isinstance(event, pd.Series):
                if event.currency == currency1:
                    self.buy(size=10000, sl=sl_price)
                elif event.currency == currency2:
                    self.sell(size=10000, sl=entry_price + self.atr_multiplier * atr_value)
            else:
                if currency1 in event["currency"].iloc[-1]:
                    self.buy(size=10000, sl=sl_price, )
                elif currency2 in event["currency"].iloc[-1]:
                    self.sell(size=10000, sl=entry_price + self.atr_multiplier * atr_value)
