import numpy as np
import pandas as pd
# import yfinance as yf
import backtrader as bt
import collections
import joblib
import backtrader.indicators as btind


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score

MAINSIGNALS = collections.OrderedDict(
    (('longshort', bt.SIGNAL_LONGSHORT),
     ('longonly', bt.SIGNAL_LONG),
     ('shortonly', bt.SIGNAL_SHORT),)
)


EXITSIGNALS = {
    'longexit': bt.SIGNAL_LONGEXIT,
    'shortexit': bt.SIGNAL_LONGEXIT,
}


class Strategy(bt.Strategy):
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        if self.dataclose[0] < self.dataclose[-1]:
            # current close less than previous close

            if self.dataclose[-1] < self.dataclose[-2]:
                # previous close less than the previous close

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.buy()

class PerformanceAnalyzer(bt.Analyzer):
    def __init__(self):
        self.returns = []
        self.win_trades = 0
        self.loss_trades = 0
        self.total_profit = 0
        self.total_loss = 0

    def notify_trade(self, trade):
        if trade.isclosed:  # Only consider fully closed trades
            pnl = trade.pnl  # Profit and loss from the trade
            self.returns.append(pnl)
            if pnl > 0:
                self.win_trades += 1
                self.total_profit += pnl
            else:
                self.loss_trades += 1
                self.total_loss += abs(pnl)

    def get_analysis(self):
        total_trades = self.win_trades + self.loss_trades
        win_rate = (self.win_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (self.total_profit / self.total_loss) if self.total_loss > 0 else 0

        return {
            'Total Trades': total_trades,
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor,
            'Total Profit': self.total_profit,
            'Total Loss': self.total_loss,
        }
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED: Price={order.executed.price:.2f}, Size={order.executed.size}")
            elif order.issell():
                print(f"SELL EXECUTED: Price={order.executed.price:.2f}, Size={order.executed.size}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print("Order Canceled/Margin/Rejected")

MAINSIGNALS = collections.OrderedDict(
    (('longshort', bt.SIGNAL_LONGSHORT),
     ('longonly', bt.SIGNAL_LONG),
     ('shortonly', bt.SIGNAL_SHORT),)
)


EXITSIGNALS = {
    'longexit': bt.SIGNAL_LONGEXIT,
    'shortexit': bt.SIGNAL_LONGEXIT,
}


class SMACloseSignal(bt.Indicator):
    lines = ('signal',)
    params = (('period', 30),)

    def __init__(self):
        self.lines.signal = self.data - bt.indicators.SMA(period=self.p.period)


class SMAExitSignal(bt.Indicator):
    lines = ('signal',)
    params = (('p1', 5), ('p2', 30),)

    def __init__(self):
        sma1 = bt.indicators.SMA(period=self.p.p1)
        sma2 = bt.indicators.SMA(period=self.p.p2)
        self.lines.signal = sma1 - sma2

# def backtest(stock):
#     cerebro = bt.Cerebro()
#     # cerebro.addstrategy(Strategy)
#     cerebro.broker.setcash(100000.0)
#     # price_data = yf.Ticker(stock).history(period="5d", interval="1m")
#     # price_data.index = pd.to_datetime(price_data.index)
#     price_data = pd.read_csv('history/EUR_CAD_M30.csv', index_col=0, header=0, parse_dates=True)
#     price_data.rename(columns={'time': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    
#     price_data = price_data.iloc[:, :5]
#     data = bt.feeds.PandasData(dataname=price_data)
#     cerebro.adddata(data)
    
#     cerebro.add_signal(bt.SIGNAL_LONGSHORT,
#                        SMACloseSignal, period=30)

#     cerebro.add_signal(bt.SIGNAL_LONGSHORT,
#                            SMAExitSignal,
#                            p1=5,
#                            p2=30)
    
#     cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=True)
#     cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
#     cerebro.addanalyzer(PerformanceAnalyzer, _name="performance")
#     print(f'Starting Portfolio Value: {cerebro.broker.getvalue()}')
#     results = cerebro.run()
#     print(f'Final Portfolio Value: {cerebro.broker.getvalue()}')
#     performance = results[0].analyzers.performance.get_analysis()
#     print(performance)
#     print("Analysis:")
#     for i in performance:
#         print(f"{i}: {performance[i]}")
#     cerebro.plot()
    
    
    
class HoldPositionStrategy(bt.Strategy):
    params = (("take_profit", 0.02),  # 2% take profit
              ("stop_loss", 0.01))   # 1% stop loss

    def __init__(self):

        self.position_price = None  # Track the entry price
        self.trade_open = False     # Flag to track if a position is active
        
        # Load the pre-trained ML model
        self.model = joblib.load("lin_reg_ml_model.pkl")


    def next(self):
        # Get the current closing price
        current_price = self.data.close[0]

        # Gather features for the current bar
        features = [
            x for x in [
                self.data._dataname['open'].iloc[0],
                # self.data._dataname['high'].iloc[0],
                # self.data._dataname['low'].iloc[0],
                # self.data._dataname['volume'].iloc[0],
                # self.data._dataname['ema_20'].iloc[0],
                # self.data._dataname['ema_50'].iloc[0],
                # self.data._dataname['ema_200'].iloc[0],
                # self.data._dataname['rsi'].iloc[0],
                # self.data._dataname['MACD_6_13_5'].iloc[0],
                # self.data._dataname['MACDh_6_13_5'].iloc[0],
                # self.data._dataname['MACDs_6_13_5'].iloc[0],
                # self.data._dataname['BBL_20_2.0'].iloc[0],
                # self.data._dataname['BBM_20_2.0'].iloc[0],
                # self.data._dataname['BBU_20_2.0'].iloc[0],
                # self.data._dataname['BBB_20_2.0'].iloc[0],
                # self.data._dataname['BBP_20_2.0'].iloc[0],
                # self.data._dataname['atr'].iloc[0],
                # self.data._dataname['obv'].iloc[0],
                # self.data._dataname['PP'].iloc[0],
                # self.data._dataname['R1'].iloc[0],
                # self.data._dataname['R2'].iloc[0],
                # self.data._dataname['R3'].iloc[0],
                # self.data._dataname['S1'].iloc[0],
                # self.data._dataname['S2'].iloc[0],
                # self.data._dataname['S3'].iloc[0]
            ] if not np.isnan(x)

        ]


        
        if not self.trade_open:
            # Generate signals (you can use your ML prediction or other logic)
            predicted_close = self.model.predict([features])[0][0] * 15
            print(predicted_close)
            print(current_price)
            if predicted_close > current_price * 1.01:  # Long signal
                self.buy(size=100)
                self.position_price = current_price
                self.trade_open = True
            elif predicted_close < current_price * 0.99:  # Short signal
                self.sell(size=100)
                self.position_price = current_price
                self.trade_open = True
        else:
            # Evaluate exit conditions
            if self.position.size > 0:  # Long position
                if current_price >= self.position_price * (1 + self.params.take_profit):
                    self.close()  # Take profit
                    print("taking profit")
                    self.trade_open = False
                elif current_price <= self.position_price * (1 - self.params.stop_loss):
                    self.close()  # Stop loss
                    print("stoploss hit")
                    self.trade_open = False
            elif self.position.size < 0:  # Short position
                if current_price <= self.position_price * (1 - self.params.take_profit):
                    self.close()  # Take profit
                    print("taking profit on short")
                    self.trade_open = False
                elif current_price >= self.position_price * (1 + self.params.stop_loss):
                    self.close()  # Stop loss
                    print("stoploss hit on short")
                    self.trade_open = False

class MLStrategy(bt.Strategy):
    def __init__(self):
        self.movav = btind.MovingAverageSimple(self.data.close, period=30)
        
        # Load the pre-trained ML model
        self.model = joblib.load("lin_reg_ml_model.pkl")

    def next(self):
        # Gather features for the current bar
        features = [
            self.movav[0],  # RSI value
        ]
        
        # Ensure no NaN values (early bars may have missing data)
        if None in features or any(np.isnan(features)):
            return
        
        # Get the current closing price
        current_close = self.data.close[0]

        # Predict using the ML model
        predicted_close = self.model.predict([features])[0][0]  # Output: 1 (buy), -1 (sell), 0 (hold)

        # Generate signals based on the prediction
        if predicted_close > current_close * 1.005:  # Buy if predicted close is 1% higher
            self.buy(size=1)
            print(f"Buy signal: Predicted = {predicted_close}, Current = {current_close}")
        elif predicted_close < current_close * 0.995:  # Sell if predicted close is 1% lower
            self.sell(size=1)
            print(f"Sell signal: Predicted = {predicted_close}, Current = {current_close}")
        else:
            print(f"Hold: Predicted = {predicted_close}, Current = {current_close}")

# 创建策略继承bt.Strategy
class BiggerThanEmaStrategy(bt.Strategy):
    params = (
        # 均线参数设置15天，15日均线
        ("maperiod", 15),
    )

    def log(self, txt, dt=None):
        # 记录策略的执行日志
        dt = dt or self.datas[0].datetime.date(0)
        print("%s, %s" % (dt.isoformat(), txt))

    def __init__(self):
        # 保存收盘价的引用
        self.dataclose = self.datas[0].close
        # 跟踪挂单
        self.order = None
        # 买入价格和手续费
        self.buyprice = None
        self.buycomm = None
        # 加入指标
        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod
        )
        # Indicators for the plotting show
        bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
        bt.indicators.WeightedMovingAverage(self.datas[0], period=25, subplot=True)
        bt.indicators.StochasticSlow(self.datas[0])
        bt.indicators.MACDHisto(self.datas[0])
        rsi = bt.indicators.RSI(self.datas[0])
        bt.indicators.SmoothedMovingAverage(rsi, period=10)
        bt.indicators.ATR(self.datas[0], plot=False)

    # 订单状态通知，买入卖出都是下单
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # broker 提交/接受了，买/卖订单则什么都不做
            return
        # 检查一个订单是否完成
        # 注意: 当资金不足时，broker会拒绝订单
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    "Bought Price: %.2f, Fee: %.2f, Commission %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log(
                    "Sell Price: %.2f, Fee: %.2f, Commission %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )
            # 记录当前交易数量
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Cancellation/Insufficient Margin/Rejection")
        # 其他状态记录为：无挂起订单
        self.order = None

    # 交易状态通知，一买一卖算交易
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log("Trading profit, gross profit %.2f, net profit %.2f" % (trade.pnl, trade.pnlcomm))

    def next(self):
        # 记录收盘价
        self.log("Close, %.2f" % self.dataclose[0])
        # 如果有订单正在挂起，不操作
        if self.order:
            return
        # 如果没有持仓则买入
        if not self.position:
            # 今天的收盘价在均线价格之上
            if self.dataclose[0] > self.sma[0]:
                # 买入
                self.log("Buy order %.2f" % self.dataclose[0])
                # 跟踪订单避免重复
                self.order = self.buy()
        else:
            # 如果已经持仓，收盘价在均线价格之下
            if self.dataclose[0] < self.sma[0]:
                # 全部卖出
                self.log("Sell Order %.2f" % self.dataclose[0])
                # 跟踪订单避免重复
                self.order = self.sell()

class LSTMPredict(bt.Strategy):
    params = (('period', 10), ('neurons', 50), ('train_size', 0.8), ('lookback', 20))

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lookback = self.p.lookback
        self.train_size = self.p.train_size
        self.train_data, self.test_data = self._prepare_data()
        # self.model = self._build_model()
        # self.model.fit(self.train_data['X'], self.train_data['Y'], epochs=50, batch_size=1, verbose=2)
        # joblib.dump(self.model, "LSTM_ml_model.pkl")
        self.model = joblib.load("LSTM_ml_model.pkl")

    def _prepare_data(self):
        data = np.array(self.dataclose)
        data = np.reshape(data, (-1, 1))
        data = self.scaler.fit_transform(data)
        train_size = int(len(data) * self.train_size)
        train_data = data[:train_size]
        test_data = data[train_size:]
        return {'X': self._prepare_X(train_data), 'Y': self._prepare_Y(train_data)}, {'X': self._prepare_X(test_data)}

    def _prepare_X(self, data):
        X, Y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i, 0])
            Y.append(data[i, 0])
        X, Y = np.array(X), np.array(Y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X

    def _prepare_Y(self, data):
        Y = []
        for i in range(self.lookback, len(data)):
            Y.append(data[i, 0])
        Y = np.array(Y)
        Y = np.reshape(Y, (Y.shape[0], 1))
        return Y

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(units=self.p.neurons, input_shape=(self.lookback, 1)))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='Adagrad', loss='mean_squared_error')
        return model

    def next(self):
        if not self.position:
            X = self.test_data['X']
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            y_pred = self.model.predict(X)
            y_pred = self.scaler.inverse_transform(y_pred)
            print('%.5f -- %.5f ' % (self.dataclose[0], y_pred[-1]))
            if self.dataclose[0] < y_pred[-1]:
                self.buy(size=100)
        else:
            if self.dataclose[0] > self.position.price:
                self.sell(size=self.position.size)



class KMeans_LinRegression_Predict(bt.Strategy):
    params = (('period', 10), ('neurons', 50), ('train_size', 0.8), ('lookback', 20))

    def __init__(self):
        # Extract data from trades
        self.prices = self.datas[0].close
        self.timestamps = self.datas[0].datetime
        
    def get_optimum_clusters(self, df, saturation_point=0.05):
        wcss = []
        k_models = []
        size = min(11, len(df.index))
        for i in range(1, size):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=None)
            kmeans.fit(df)
            wcss.append(kmeans.inertia_)
            k_models.append(kmeans)

        #print(wcss)
        #print(k_models)

        #           SILOUETTE METHOD
        optimum_k = len(wcss) - 1
        for i in range(0, len(wcss) - 1):
            diff = abs(wcss[i + 1] - wcss[i])
            if diff < saturation_point:
                optimum_k = i
                break

        print("Optimum K is " + str(optimum_k + 1))
        optimum_clusters = k_models[optimum_k]

        return optimum_clusters

    def calculate_suuport_resistance(self, gran, ticker, start_date, end_date):

        lows = pd.DataFrame(data=data, index=data.index, columns=["Close"])
        #highs = pd.DataFrame(data=data, index=data.index, columns=["High"])

        low_clusters = self.get_optimum_clusters(lows)
        low_centers = low_clusters.cluster_centers_
        low_centers = np.sort(low_centers, axis=0)

        high_clusters = self.get_optimum_clusters(highs)
        high_centers = high_clusters.cluster_centers_
        high_centers = np.sort(high_centers, axis=0)

        lowss = []
        highss = []

        for i in low_centers:
            i = float(i)
            lowss.append(i)

        for i in high_centers:
           i = float(i)
           highss.append(i)

        print('lows/support: ', lowss)
        #print('highs/resistance: ', highss)

        return lowss, highss, data

    def next(self):
        current_price = self.datas[0].close[0]

        # Perform linear regression on the prices
        x = np.array(self.timestamps).reshape(-1, 1)
        y = np.array(self.prices).reshape(-1, 1)
        reg = LinearRegression().fit(x, y)

        # Initializing regression
        reg_value = round(float(reg.predict(np.array([[self.timestamps[-1]]]))[0][0]), 5)

        # Fit KMeans model
        kmeans = KMeans(n_clusters=2, n_init='auto')
        kmeans.fit(np.array(self.prices).reshape(-1, 1))

        # Get cluster centers
        centers = sorted(kmeans.cluster_centers_.ravel())

        # Set lower and upper resistance levels as the midpoint between the cluster centers
        self.lower_resistance = round(centers[0], 5)
        self.upper_resistance = round(centers[1], 5)
        
        mean_buy = 0
        deducted = 0


def test():
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    price_data = pd.read_csv('history/EUR_CAD_H4.csv', index_col=0, header=0, parse_dates=True)
    price_data.rename(columns={'time': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    
    # price_data = price_data.iloc[:, :5]
    data = bt.feeds.PandasData(dataname=price_data)
    cerebro.adddata(data)
    cerebro.addstrategy(KMeans_LinRegression_Predict)

    
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(PerformanceAnalyzer, _name="performance")

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue()}')
    results = cerebro.run()
    print(f'Final Portfolio Value: {cerebro.broker.getvalue()}')
    performance = results[0].analyzers.performance.get_analysis()

    print(performance)
    print(results[0].analyzers.sharpe.get_analysis())

    print(results[0].analyzers.drawdown.get_analysis())
    print("Analysis:")
    for i in performance:
        print(f"{i}: {performance[i]}")
    cerebro.plot()
    
    
test()

# stock = "TSLA"


# backtest(stock)