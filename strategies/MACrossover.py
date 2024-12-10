from backtesting import Strategy
from backtesting.lib import crossover

class MACrossover(Strategy):
    n1 = 10
    n2 = 30
    
    @staticmethod
    def get_optimization_params():
        return dict(
            n1=range(4, 20, 2),
            n2=range(20, 50, 5),
        )

    def init(self):
        self.ma1 = self.data.Close.rolling(self.n1).mean()
        self.ma2 = self.data.Close.rolling(self.n2).mean()

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()
