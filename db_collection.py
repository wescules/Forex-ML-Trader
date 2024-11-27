"""
script that runs continuously and collects pricing and technical indicators data,
entering the data into instrument/window data tables.
"""
import pandas as pd
from backend.config import currencies
from backend.indicators import pivotPoints
import pandas_ta as ta
import tpqoa

WINDOWS = ['M5', 'M15', 'M30', 'H1', 'H4', 'D', 'W']

def liteCurrencyDB(instrument, granularity, save_csv=False):
    # get data
    df = download_history(instrument, granularity)
    
    df.drop_duplicates(inplace=True)

    # set up indicators
    df["ema_20"] = ta.ema(df["Close"], length=20)
    df["ema_50"] = ta.ema(df["Close"], length=50)
    df["ema_200"] = ta.ema(df["Close"], length=200)

    window_index = WINDOWS.index(granularity)
    rsi_settings = [10, 14, 18, 21, 25, 30, 35]
    df["rsi"] = ta.rsi(df["Close"], length=rsi_settings[window_index])

    macd_settings = [[6, 13, 5], [12, 26, 9], [18, 35, 12], [
        24, 48, 18], [36, 72, 24], [48, 96, 36], [48, 96, 36]]
    macd = ta.macd(df["Close"], fast=macd_settings[window_index][1],
                   slow=macd_settings[window_index][0], signal=macd_settings[window_index][2])
    df = pd.concat([df, macd], axis=1)

    bbands = ta.bbands(df["Close"], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)

    # https://www.investopedia.com/terms/a/atr.asp
    atr_settings = [14, 14, 14, 21, 21, 28, 28]
    df["atr"] = ta.atr(df["High"], df["Low"], df["Close"],
                       length=atr_settings[window_index])

    # https://www.investopedia.com/terms/o/onbalancevolume.asp
    df["obv"] = ta.obv(df["Close"], df["Volume"])

    pp = pd.DataFrame(data=pivotPoints(df))
    df = pd.concat([df, pp], axis=1)


    # collect data, useful for research and weekends/holidays when the market isn't open
    if save_csv:
        title = 'history/%s_%s.csv' % (instrument, granularity)
        df.to_csv(title)

    # connect to the DB
    # db = sqlite3.connect('liteDB/currencies.db')
    # c = db.cursor()
    # table_name = "%s_%s" % (instr, w)
    # inputs = list(tuple(row for idx, row in df.iterrows()))
    # insert them into the dedicated table for this currency and timeframe
    # c.execute('''CREATE TABLE IF NOT EXISTS {}(
    #             dt TEXT, Open REAL, High REAL, Low REAL, Close REAL,
    #             Volume INTEGER, fast_sma REAL, slow_sma REAL, Lstoch REAL,
    #             Hstoch REAL, K REAL, D REAL, rolling_mean REAL, boll_high REAL,
    #             boll_low REAL, PP REAL, R1 REAL, R2 REAL, R3 REAL, S1 REAL,
    #             S2 REAL, S3 REAL
    #             )'''.format(table_name))
    # c.executemany('''INSERT INTO {}(
    #         dt, Open, High, Low, Close, Volume, fast_sma, slow_sma, Lstoch,
    #         Hstoch, K, D, rolling_mean, boll_high, boll_low, PP, R1, R2, R3,
    #         S1, S2, S3
    #         ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''
    #         .format(table_name), inputs)
    # db.commit()
    # print('table created successfully')
    # return db, df


def download_history(instrument, window, start="2024-01-05", end="2024-11-26"):
    print("Downloading %s -- %s historical data..." % (instrument, window))

    oanda = tpqoa.tpqoa("oanda.cfg")

    df = oanda.get_history(
        instrument, start, end, window, "M"
    )
    
    df.rename(columns={'time': 'dt', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'volume': 'Volume'}, inplace=True)
    df.drop("complete", axis=1, inplace=True)
    print("Download complete.")

    return df

if __name__ == "__main__":

    for instrument in currencies:
        for window in WINDOWS:
            liteCurrencyDB(instrument, window, save_csv=True)
