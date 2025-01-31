import pandas as pd

def generate_signals(data):
    data['short_ma'] = data['price'].rolling(window=5).mean()
    data['long_ma'] = data['price'].rolling(window=15).mean()

    data['position'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'position'] = 1
    data.loc[data['short_ma'] < data['long_ma'], 'position'] = -1

    return data[['position']]
