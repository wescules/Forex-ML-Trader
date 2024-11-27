# price history or stream functions

from .config import token, accountID, env
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.pricing import PricingStream
import pandas as pd
import tpqoa

api = API(token)

# returns the last 500 OHLCV candles for an instrument
# maximum count is 500
# window: M1, M5, M15, H, H4, D, etc.
def history(instrument, window, start="2024-05-05", end="2024-11-26", collection=False):
    print("Downloading historical data...")

    oanda = tpqoa.tpqoa("oanda.cfg")

    df = oanda.get_history(
        instrument, start, end, window, "M"
    )
    
    df.rename(columns={'time': 'dt', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'volume': 'Volume'}, inplace=True)
    df.drop("complete", axis=1, inplace=True)
    print("Download complete.")
    # collect data, useful for research and weekends/holidays when the market isn't open
    if collection == True:
        title = 'data/%s_%s_history.csv' % (instrument, window)
        df.to_csv(title)

    return df

# creates a pricing stream,
def stream(instrument, window):
    request_params = {"timeout":100}
    params = {"instruments":instrument, "granularity":window}
    api = API(access_token=token,environment=env, request_params=request_params)
    r = PricingStream(accountID=accountID, params=params)

    while True:
        try:
            api.request(r)
            for R in r.response:
                time = R['time']
                ask = R['asks'][0]['price']
                bid = R['bids'][0]['price']
                return time, ask, bid
        except Exception as e:
            print(e)
            continue
