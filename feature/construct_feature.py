import pandas as pd
from talib.abstract import *
from target.valid_wave.valid_wave import tag_wave_direction


def construct_features1(ohlcv, params, test=False):
    data = pd.DataFrame(index=ohlcv.index)
    data['open_close'] = (ohlcv['open'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['high_close'] = (ohlcv['high'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['low_close'] = (ohlcv['low'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['pct_chg'] = ((ohlcv['close'] / ohlcv['close'].shift(1)) - 1) * 100
    data['std'] = data['pct_chg'].rolling(params['std_window']).std().bfill()
    data = data.fillna(0)

    ma_volume = ohlcv['volume'].rolling(params['vol_window']).mean()
    data['volume'] = ohlcv['volume'] / ma_volume
    data['volume'] = data['volume'].fillna(1)

    next_ma = SMA(ohlcv, timeperiod=params['ma']).shift(-params['ma'])
    data['label'] = 1
    data.loc[ohlcv['close'] > next_ma, 'label'] = 0
    data.loc[ohlcv['close'] < next_ma, 'label'] = 2

    if test:
        return pd.concat([data, ohlcv], axis=1)
    else:
        return data


def construct_features2(ohlcv, params, test=False):
    data = pd.DataFrame(index=ohlcv.index)
    data['open_close'] = (ohlcv['open'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['high_close'] = (ohlcv['high'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['low_close'] = (ohlcv['low'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['pct_chg'] = ((ohlcv['close'] / ohlcv['close'].shift(1)) - 1) * 100
    data['std'] = data['pct_chg'].rolling(params['std_window']).std().bfill()
    price_std = ohlcv['close'].rolling(params['std_window']).std().bfill()
    data = data.fillna(0)

    ma_volume = ohlcv['volume'].rolling(params['vol_window']).mean()
    data['volume'] = ohlcv['volume'] / ma_volume
    data['volume'] = data['volume'].fillna(1)

    next_ma = SMA(ohlcv, timeperiod=params['ma']).shift(-params['ma'])
    data['label'] = 1
    data.loc[ohlcv['close'] > next_ma + (params['n_std'] * price_std), 'label'] = 0
    data.loc[ohlcv['close'] < next_ma - (params['n_std'] * price_std), 'label'] = 2

    if test:
        return pd.concat([data, ohlcv], axis=1)
    else:
        return data


def construct_features3(ohlcv, params, test=False):
    data = pd.DataFrame(index=ohlcv.index)
    data['open_close'] = (ohlcv['open'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['high_close'] = (ohlcv['high'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['low_close'] = (ohlcv['low'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)

    data['pct_chg'] = ((ohlcv['close'] / ohlcv['close'].shift(1)) - 1) * 100
    data['std'] = data['pct_chg'].rolling(params['std_window']).std().bfill()
    data = data.fillna(0)

    ma_volume = ohlcv['volume'].rolling(params['vol_window']).mean()
    data['volume'] = ohlcv['volume'] / ma_volume
    data['volume'] = data['volume'].fillna(1)

    new_ohlcv = tag_wave_direction(ohlcv, params['max_return_threshold'], params['return_per_count_threshold'],
                                   params['withdraw_threshold'], params['minimum_period'], operation='label',
                                   mode='relative', std_window=params['std_window'])
    direction = new_ohlcv['direction']

    data['label'] = direction.fillna(0) + 1

    if test:
            return pd.concat([data, new_ohlcv], axis=1)
    else:
            return data
