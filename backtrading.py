import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=FutureWarning, module="ta.trend")

# Custom Analyzer for tracking performance and portfolio value
class PerformanceAnalyzer(bt.Analyzer):
    def __init__(self):
        self.portfolio_values = []
        self.pnl = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.pnl.append(trade.pnl)

    def next(self):
        # Track portfolio value at each step
        self.portfolio_values.append(self.strategy.broker.get_value())

    def get_analysis(self):
        return {
            'cumulative_pnl': sum(self.pnl),
            'average_pnl': np.mean(self.pnl) if self.pnl else 0,
            'max_pnl': max(self.pnl) if self.pnl else 0,
            'min_pnl': min(self.pnl) if self.pnl else 0,
            'portfolio_values': self.portfolio_values
        }

# Define a custom indicator to track the target value for the AI strategy
class TargetIndicator(bt.Indicator):
    lines = ('target',)

    def __init__(self):
        pass

    def next(self):
        # Target value for upward trend (1) or downward (0)
        if self.data.close[0] > self.data.close[-1]:
            self.lines.target[0] = 1
        else:
            self.lines.target[0] = 0

# AI Reversal Strategy
class AIReversalStrategy(bt.Strategy):
    params = (
        ('symbol', 'BTCUSD'),
        ('interval', 5),  # Interval in minutes
    )

    def __init__(self):
        self.signals = None
        self.order = None
        self.buy_signal = None
        self.sell_signal = None
        self.buy_signals = []
        self.sell_signals = []

        # Initialize buy/sell signal prices to track executed prices
        self.buy_signal_price = None
        self.sell_signal_price = None

        # Technical indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.sma_10 = bt.indicators.SimpleMovingAverage(self.data.close, period=10)
        self.sma_30 = bt.indicators.SimpleMovingAverage(self.data.close, period=30)
        self.stoch_k = bt.indicators.StochasticSlow(self.data)
        self.macd = bt.indicators.MACD(self.data.close)

        # Custom target indicator (up/down trend)
        self.target = TargetIndicator(self.data)

        # Variables for machine learning
        self.model = None
        self.signals = None

    def next(self):
        current_time = datetime.now()
        # if current_time.weekday() >= 5:  # Skip weekends
        #     return

        # if current_time.minute % self.params.interval == 0:
        self.signals = self.run_ai_reversal_pipeline(self.data_raw)
        self.execute_orders()

        # Display target value
        print(f"Target: {self.target.target[0]}")

        # AI prediction logic
        if self.model is not None:
            features = self.prepare_features()
            prediction = self.model.predict([features])
            signal = 'Buy' if prediction == 1 else 'Sell'
            print(f"AI Prediction: {signal}")

            # Buy Signal
            if signal == 'Buy' and not self.position:  # Buy only if no current position
                print("Generating Buy signal...")
                self.buy_signal_price = self.data.close[0]
                self.order = self.buy()  # Execute buy order
                print(f"Buy signal executed at price: {self.buy_signal_price}")
                self.buy_signals.append((self.data.datetime.datetime(), self.data.close[0]))

            # Sell Signal
            elif signal == 'Sell' and self.position:  # Sell only if currently holding a position
                print("Generating Sell signal...")
                self.sell_signal_price = self.data.close[0]
                self.order = self.sell()  # Execute sell order
                print(f"Sell signal executed at price: {self.sell_signal_price}")
                self.sell_signals.append((self.data.datetime.datetime(), self.data.close[0]))

    def prepare_features(self):
        # Convert Backtrader LineBuffer objects to Pandas Series for ML model prediction
        close_series = pd.Series(self.data.close.get(size=len(self.data)), index=self.data.datetime.get(size=len(self.data)))
        rsi_series = pd.Series(self.rsi.get(size=len(self.rsi)), index=self.data.datetime.get(size=len(self.rsi)))
        sma_10_series = pd.Series(self.sma_10.get(size=len(self.sma_10)), index=self.data.datetime.get(size=len(self.sma_10)))
        sma_30_series = pd.Series(self.sma_30.get(size=len(self.sma_30)), index=self.data.datetime.get(size=len(self.sma_30)))
        stoch_k_series = pd.Series(self.stoch_k.get(size=len(self.stoch_k)), index=self.data.datetime.get(size=len(self.stoch_k)))
        macd_series = pd.Series(self.macd.get(size=len(self.macd)), index=self.data.datetime.get(size(len(self.macd))))

        # Prepare feature vector (ignoring NA values)
        features = pd.DataFrame({
            'rsi': rsi_series,
            'sma_10': sma_10_series,
            'sma_30': sma_30_series,
            'stoch_k': stoch_k_series,
            'macd': macd_series
        }).dropna().iloc[-1].values  # Get the most recent row for prediction

        return features

    def run_ai_reversal_pipeline(self, data):
        if len(data) == 0:
            print("Error: DataFrame is empty. Skipping AI reversal pipeline.")
            return None

        print("Running AI reversal pipeline...")
        prepared_data = self.prepare_data(data)

        features, target = self.feature_engineering(prepared_data)
        model = self.train_and_evaluate_model(features, target)
        self.model = model
        print("Model training completed.")

        return prepared_data

    def prepare_data(self, data):
        close_series = pd.Series(data.close.get(size=len(data)), index=data.datetime.get(size(len(data))))
        data['target'] = np.where(close_series.shift(-1) > close_series, 1, 0)
        return data.dropna()

    def feature_engineering(self, data):
        features = data[['rsi', 'sma_10', 'sma_30', 'stoch_k', 'macd']]
        target = data['target']
        return features, target

    def tune_model(self, features, target):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(features, target)
        return grid_search.best_estimator_

    def train_and_evaluate_model(self, features, target):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = self.tune_model(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
        return model

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED: Price = {order.executed.price}, Size = {order.executed.size}")
            elif order.issell():
                print(f"SELL EXECUTED: Price = {order.executed.price}, Size = {order.executed.size}")
            self.order = None  # Reset the order variable

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print("Order Canceled/Margin/Rejected")
            self.order = None  # Reset the order variable

    def notify_trade(self, trade):
        if trade.isclosed:
            print(f"TRADE CLOSED: Gross PnL = {trade.pnl:.2f}, Net PnL = {trade.pnlcomm:.2f}")

# Initialize Cerebro engine
cerebro = bt.Cerebro()

# Add strategy to Cerebro
cerebro.addstrategy(AIReversalStrategy)

# Load Data (example CSV file with timestamp, open, high, low, close, volume columns)
file = 'history/AUD_CAD_H1.csv'

data = pd.read_csv(file, parse_dates=['time'])
data.set_index('time', inplace=True)
datafeed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(datafeed)

# Configure the broker settings
cerebro.broker.set_cash(1000)
cerebro.broker.setcommission(commission=0.001)

# Add analyzers to evaluate strategy performance
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

# Add observers to plot buy/sell signals and portfolio changes
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Broker)

# Run the strategy
results = cerebro.run()
final_value = cerebro.broker.get_value()
print(f"Final Portfolio Value: {final_value}")

# Plot the results (including buy/sell signals and portfolio changes)
cerebro.plot(style='candlestick')
