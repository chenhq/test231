import pandas as pd
from talib.abstract import *
import numpy as np
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


# params = {
#     'window': 60,
# }
def features_kline(ohlcv, params, test=False):
    data = pd.DataFrame(index=ohlcv.index)

    open_close = (ohlcv['open'] - ohlcv['close']) / ohlcv['close']
    col = 'open_close_{}'.format(params['window'])
    data[col] = (open_close - open_close.rolling(params['window']).mean().bfill()) / open_close.rolling(
        params['window']).std().bfill()

    high_close = (ohlcv['high'] - ohlcv['close']) / ohlcv['close']
    col = 'high_close_{}'.format(params['window'])
    data[col] = (high_close - high_close.rolling(params['window']).mean().bfill()) / high_close.rolling(
        params['window']).std().bfill()

    low_close = (ohlcv['low'] - ohlcv['close']) / ohlcv['close']
    col = 'low_close_{}'.format(params['window'])
    data[col] = (low_close - low_close.rolling(params['window']).mean().bfill()) / low_close.rolling(
        params['window']).std().bfill()

    pct_chg = ohlcv['close'].pct_change().fillna(0)
    col = 'pct_chg_{}'.format(params['window'])
    data[col] = (low_close - pct_chg.rolling(params['window']).mean().bfill()) / pct_chg.rolling(
        params['window']).std().bfill()

    col = 'std_pct_chg_{}'.format(params['window'])
    data[col] = pct_chg.rolling(params['window']).std().bfill()

    col = 'volume_{}'.format(params['window'])
    data[col] = (ohlcv['volume'] - ohlcv['volume'].rolling(params['window']).mean().bfill()) / ohlcv[
        'volume'].rolling(params['window']).std().bfill()

    if test:
        return pd.concat([data, ohlcv], axis=1)
    else:
        return data


# params = {
#     'window': 250,
#     'next_ma_window': 3,
#     'quantile_list': [0, 0.1, 0.3, 0.7, 0.9, 1]
# }
def label_by_ma_price(ohlcv, params, test=False, epsilon=0.0001):
    label = pd.DataFrame(index=ohlcv.index)
    next_ma = SMA(ohlcv, timeperiod=params['next_ma_window']).shift(-params['next_ma_window'])
    price_gap = (next_ma - ohlcv['close']).fillna(0)

    quantile_list = params['quantile_list']
    for i in range(len(quantile_list)-1):
        cond = (price_gap > price_gap.rolling(params['window']).quantile(quantile_list[i]).bfill() - epsilon) & \
               (price_gap < price_gap.rolling(params['window']).quantile(quantile_list[i+1]).bfill() + epsilon)
        label.loc[cond, 'label'] = i

    if test:
        return pd.concat([label, ohlcv], axis=1)
    else:
        return label


def construct_features(ohlcv, params_list, func_list, test=False):
    result_list = []
    for i in range(len(params_list)):
        params = params_list[i]
        function = func_list[i]
        result_list.append(function(ohlcv, params))

    results = pd.concat(result_list, axis=1)

    if test:
        return pd.concat([results, ohlcv], axis=1)
    else:
        return results

