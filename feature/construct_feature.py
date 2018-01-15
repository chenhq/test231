import pandas as pd
from talib.abstract import *
import numpy as np
from target.valid_wave.valid_wave import tag_wave_direction
from data_prepare import categorical_factory
import matplotlib.pyplot as plt


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
def feature_kline(ohlcv, params, test=False):
    data = pd.DataFrame(index=ohlcv.index)

    open_close = (ohlcv['open'] - ohlcv['close']) / ohlcv['close']
    high_close = (ohlcv['high'] - ohlcv['close']) / ohlcv['close']
    low_close = (ohlcv['low'] - ohlcv['close']) / ohlcv['close']
    pct_chg = ohlcv['close'].pct_change().fillna(0)

    for window in params['window']:
        col = 'open_close_{}'.format(window)
        data[col] = (open_close - open_close.rolling(window).mean().bfill()) / open_close.rolling(
            window).std().bfill()

        col = 'high_close_{}'.format(window)
        data[col] = (high_close - high_close.rolling(window).mean().bfill()) / high_close.rolling(
            window).std().bfill()

        col = 'low_close_{}'.format(window)
        data[col] = (low_close - low_close.rolling(window).mean().bfill()) / low_close.rolling(
            window).std().bfill()

        col = 'pct_chg_{}'.format(window)
        data[col] = (pct_chg - pct_chg.rolling(window).mean().bfill()) / pct_chg.rolling(
            window).std().bfill()

        col = 'volume_{}'.format(window)
        volume_ratio = np.log(ohlcv['volume'] / ohlcv['volume'].rolling(window).mean().bfill())
        volume_mean = volume_ratio.rolling(window).mean().bfill()
        volume_std = volume_ratio.rolling(window).std().bfill()
        volume_max = volume_mean + 3 * volume_std
        volume_min = volume_mean - 3 * volume_std
        if len(volume_ratio[volume_ratio > volume_max]) > 0:
            volume_ratio[volume_ratio > volume_max] = volume_max
        if len(volume_ratio[volume_ratio < volume_min]) > 0:
            volume_ratio[volume_ratio < volume_min] = volume_min

        norm_volume = (volume_ratio - volume_mean) / volume_std
        data[col] = norm_volume

    if test:
        return pd.concat([data, ohlcv], axis=1)
    else:
        return data


def scale_max_min(series, window):
    mean_ = series.rolling(window).mean()
    max_ = series.rolling(window).max()
    min_ = series.rolling(window).min()
    for i in range(window, len(series)):
        if i % window != 0:
            mean_.iloc[i] = np.nan
            max_.iloc[i] = np.nan
            min_.iloc[i] = np.nan
    mean_ = mean_.bfill().ffill()
    max_ = max_.bfill().ffill()
    min_ = min_.bfill().ffill()
    result = (series - mean_) * 2 / (max_ - min_)
    return result


# params = {
#     'window': 256,
# }
def feature_kline2(ohlcv, params, test=False):
    window = params['window']
    close_by_window = scale_max_min(ohlcv['close'], window)
    close_by_window.name = 'close_by_window'

    volume_by_window = scale_max_min(ohlcv['volume'], window)
    volume_by_window.name = 'volume_by_window'

    pre_close = ohlcv['close'].shift(1)

    high_close = (ohlcv['high'] - ohlcv['close']) * 10 / pre_close
    high_close.name = 'high_close'

    close_low = (ohlcv['close'] - ohlcv['low']) * 10 / pre_close
    close_low.name = 'close_low'

    high_open = (ohlcv['high'] - ohlcv['open']) * 10 / pre_close
    high_open.name = 'high_open'

    open_low = (ohlcv['open'] - ohlcv['low']) * 10 / pre_close
    open_low.name = 'open_low'

    open_close_max = ohlcv[['open', 'close']].max(axis=1)
    open_close_min = ohlcv[['open', 'close']].min(axis=1)
    up = (ohlcv['high'] - open_close_max) * 10 / pre_close
    up.name = 'up'
    down = (open_close_min - ohlcv['low']) * 10 / pre_close
    down.name = 'down'
    wide = (ohlcv['open'] - ohlcv['close']).abs() * 10 / pre_close
    wide.name = 'wide'

    open_ = (ohlcv['open'] - pre_close) * 10 / pre_close
    open_.name = 'open_'
    high_ = (ohlcv['high'] - pre_close) * 10 / pre_close
    high_.name = 'high_'
    low_ = (ohlcv['low'] - pre_close) * 10 / pre_close
    low_.name = 'low_'
    close_ = (ohlcv['close'] - pre_close) * 10 / pre_close
    close_.name = 'close_'
    up_down = (ohlcv['close'] - pre_close).map(np.sign)
    up_down.name = 'up_down'

    diff_high_close_low = high_close - close_low
    diff_high_close_low.name = 'diff_high_close_low'
    diff_high_open_low = high_open - open_low
    diff_high_open_low.name = 'diff_high_open_low'

    diff_up_down = up - down
    diff_up_down.name = 'diff_up_down'
    diff_up_wide = up - wide
    diff_up_wide.name = 'diff_up_wide'
    diff_down_wide = down - wide
    diff_down_wide.name = 'diff_down_wide'

    rsi = RSI(ohlcv, timeperiod=14).bfill()
    rsi.name = 'rsi'
    rsi /= 100
    macd = MACD(ohlcv).bfill()
    macd /= 200.0
    stoch = STOCH(ohlcv).bfill()
    k, d = stoch['slowk'], stoch['slowd']
    j = 3 * k - 2 * d
    j.name = 'j'

    k = (k - 50) / 50
    d = (d - 50) / 50
    j = (j - 50) / 50
    j_k = j - k
    j_k.name = 'j_k'
    j_d = j - d
    j_d.name = 'j_d'
    k_d = k - d
    k_d.name = 'k_d'

    # bbands
    bbands = BBANDS(ohlcv)
    close_to_upperband = ohlcv['close'] - bbands['upperband']
    close_to_upperband.name = 'close_to_upperband'
    close_to_upperband /= 250
    close_to_middleband = ohlcv['close'] - bbands['middleband']
    close_to_middleband.name = 'close_to_middleband'
    close_to_middleband /= 250
    close_to_lowerband = ohlcv['close'] - bbands['lowerband']
    close_to_lowerband.name = 'close_to_lowerband'
    close_to_lowerband /= 250

    upper_to_middle = bbands['upperband'] - bbands['middleband']
    upper_to_middle.name = 'upper_to_middle'
    upper_to_middle /= 250

    lower_to_middle = bbands['lowerband'] - bbands['middleband']
    lower_to_middle.name = 'lower_to_middle'
    lower_to_middle /= 250

    close_upper_middle_ratio = close_to_middleband / upper_to_middle
    close_upper_middle_ratio.name = 'close_upper_middle_ratio'
    close_lower_middle_ratio = close_to_middleband / lower_to_middle
    close_lower_middle_ratio.name = 'close_lower_middle_ratio'

    new_bbands = pd.concat([close_to_upperband, close_to_middleband, close_to_lowerband, upper_to_middle,
                            lower_to_middle, close_upper_middle_ratio, close_lower_middle_ratio], axis=1)

    ma20 = MA(ohlcv, timeperiod=20).bfill()
    period_idx = (ma20 - ma20.shift(11)) / ma20.shift(11)
    period_idx.name = 'period_idx'

    # ma3 = MA(ohlcv, timeperiod=3)
    # ma5 = MA(ohlcv, timeperiod=5)
    # ma10 = MA(ohlcv, timeperiod=10)
    # ma15 = MA(ohlcv, timeperiod=15)
    # ma30 = MA(ohlcv, timeperiod=30)
    # ma60 = MA(ohlcv, timeperiod=60)
    # ma240 = MA(ohlcv, timeperiod=240)
    #
    # ma5_30 = ma5 - ma30
    # ma5_60 = ma5 - ma60
    # ma5_240 = ma5 - ma240
    # ma30_60 = ma30 - ma60
    # ma30_240 = ma30 - ma240
    # ma60_240 = ma60 - ma240

    pct_chg = ohlcv['close'].pct_change().fillna(0)
    pct_chg.name = 'pct_chg'
    pct_chg *= 10

    features_list = [close_by_window, volume_by_window, open_, high_, low_, close_, up_down,
                     high_close, close_low, high_open, open_low,
                     diff_high_close_low, diff_high_open_low,
                     up, down, wide, diff_up_down, diff_up_wide, diff_down_wide,
                     rsi, macd, k, d, j, j_k, j_d, k_d, new_bbands, period_idx, pct_chg]

    features = pd.concat(features_list, axis=1)
    features = features.fillna(0)

    if test:
        return pd.concat([features, ohlcv], axis=1)
    else:
        return features


# params = {
#     'window': 250,
#     'next_ma_window': 3,
#     'quantile_list': [0, 0.1, 0.3, 0.7, 0.9, 1]
# }
def label_by_ma_price(ohlcv, params, test=False, epsilon=0.000001):
    label = pd.DataFrame(index=ohlcv.index)
    next_ma = SMA(ohlcv, timeperiod=params['next_ma_window']).shift(-params['next_ma_window'])
    price_gap = (next_ma - ohlcv['close']).fillna(0)

    quantile_list = params['quantile_list']
    for i in range(len(quantile_list) - 1):
        cond = (price_gap > price_gap.rolling(params['window']).quantile(quantile_list[i]).bfill() - epsilon) & \
               (price_gap < price_gap.rolling(params['window']).quantile(quantile_list[i + 1]).bfill() + epsilon)
        label.loc[cond, 'label'] = i

    class_list = [i for i in range(len(params['quantile_list']) - 1)]
    to_categorical, _ = categorical_factory(class_list)
    label['label'] = label['label'].copy().map(to_categorical)
    if test:
        return pd.concat([label, ohlcv], axis=1)
    else:
        return label


# params = {
#     'window': [3, 5, 10]
# }
def label_by_multi_ma(ohlcv, params, test=False):
    up_down = ohlcv['close'].pct_change().map(np.sign).fillna(0)
    close = ohlcv['close']
    win1, win2, win3 = params['window']
    ma1 = MA(ohlcv, timeperiod=win1).shift(-win1)
    ma2 = MA(ohlcv, timeperiod=win2).shift(-win2)
    ma3 = MA(ohlcv, timeperiod=win3).shift(-win3)

    result = pd.Series(index=ohlcv.index)
    result.name = 'label'
    i = 0
    while i < len(ohlcv):
        if up_down.iloc[i] != 0:
            direction = up_down.iloc[i]
        else:
            if close.iloc[i] != ma1.iloc[i]:
                direction = np.sign(ma1.iloc[i] - close.iloc[i])
            elif close.iloc[i] != ma2.iloc[i]:
                direction = np.sign(ma2.iloc[i] - close.iloc[i])
            elif close.iloc[i] != ma3.iloc[i]:
                direction = np.sign(ma2.iloc[i] - close.iloc[i])
            else:
                direction = 1

        j = i + 1
        while j < len(ohlcv):
            if up_down.iloc[j] == direction or up_down.iloc[j] == 0:
                j += 1
            elif (ma1[j] - close.iloc[j]) * direction > 0 or (ma2[j] - close.iloc[j]) * direction > 0 or (
                ma3[j] - close.iloc[j]) * direction > 0:
                j += 1
            else:
                break

        result.iloc[range(i, j)] = direction
        i = j
    result = result.fillna(0)
    result[result == -1] = 0.0
    class_list = [0.0, 1.0]
    to_categorical, _ = categorical_factory(class_list)
    categorical_result = result.copy().map(to_categorical)
    if test:
        return pd.concat([categorical_result, ohlcv], axis=1)
    else:
        return categorical_result


def feature_pct_chg(ohlcv, price='close'):
    pct_changes = ohlcv[price].pct_change()
    pct_changes = pct_changes.fillna(0)
    return pct_changes


# params = {
#     'ma_list': [1, 2, 3, 5, 8, 13, 21, 34, 55],
#     'window': 128,
#     'price': 'close'
# }
def feature_ma(ohlcv, params, test=False):
    if 'ma_list' in params:
        ma_list = params['ma_list']
    else:
        ma_list = [1, 2, 3, 5, 8, 13, 21, 34, 55]

    if 'window' in params:
        windows = params['window']
    else:
        window = 128

    if 'price' in params:
        price = params['price']
    else:
        price = 'close'

    mas = pd.DataFrame(index=ohlcv.index)

    for win in ma_list:
        if win == 1:
            mas['ma_{}'.format(win)] = ohlcv[price]
        else:
            mas['ma_{}'.format(win)] = SMA(ohlcv, timeperiod=win, price=price).bfill()

    standard_mas = pd.DataFrame(index=ohlcv.index)
    for window in windows:
        price_max = ohlcv[price].rolling(window).max().bfill()
        price_min = ohlcv[price].rolling(window).min().bfill()

        for col in mas.columns:
            standard_mas['{}_{}'.format(col, window)] = (((mas[col] - price_min) / (price_max - price_min)) - 0.5) * 2

    if test:
        return pd.concat([standard_mas, ohlcv], axis=1)
    else:
        return standard_mas


def construct_features(ohlcv, params_list, func_list, test=False):
    result_list = []
    for i in range(len(params_list)):
        params = params_list[i]
        function = func_list[i]
        result_list.append(function(ohlcv, params))

    results = pd.concat(result_list, axis=1)

    if 'pct_chg' not in results.columns:
        results['pct_chg'] = feature_pct_chg(ohlcv, price='close')

    if test:
        return pd.concat([results, ohlcv], axis=1)
    else:
        return results
