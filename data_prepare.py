import numpy as np
import pandas as pd
from talib.abstract import *
import random
import empyrical
import multiprocessing


def construct_features(ohlcv, construct_features_func):
    return construct_features_func(ohlcv)


def construct_features_for_stocks(ohlcv_list, construct_features_func, processes=3):
    stock_features_list = []

    if processes <= 0:
        processes = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=processes)

    def callback(result):
        stock_features_list.append(result)

    def print_error(err):
        print(err)

    for ohlcv in ohlcv_list:
        pool.apply_async(construct_features_func, args=(ohlcv,), callback=callback, error_callback=print_error)

    pool.close()
    pool.join()
    return stock_features_list


def construct_features1(ohlcv):
    data = pd.DataFrame(index=ohlcv.index)
    data['open_close'] = (ohlcv['open'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['high_close'] = (ohlcv['high'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['low_close'] = (ohlcv['low'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['pct_chg'] = ((ohlcv['close'] / ohlcv['close'].shift(1)) - 1) * 100
    data = data.fillna(0)

    next_ma5 = SMA(ohlcv, timeperiod=5).shift(-5)
    data['label'] = 1
    data.loc[ohlcv['close'] < next_ma5, 'label'] = 0
    data.loc[ohlcv['close'] > next_ma5, 'label'] = 2

    ma15_volume = ohlcv['volume'].rolling(15).mean()
    data['volume'] = ohlcv['volume'] / ma15_volume
    data['volume'] = data['volume'].fillna(1)

    return data


def categorical_func_factory(num_class, class_list):
    def categorical_func(x):
        cls = []
        for i in range(num_class):
            if class_list[i] == x:
                cls.append(1.0)
            else:
                cls.append(0.0)
        return cls
    return categorical_func


def to_categorical(data, column, func):
    class_list = data[column].unique()
    class_list.sort()
    num_class = len(class_list)
    data[column] = data[column].copy().map(func(num_class, class_list))

    def reverse_func(array):
        return class_list[np.argmax(array)]

    return data, reverse_func


def split_data_by_sample(data, split_dict, minimum_size):
    result = {}
    length = len(data)
    all = 0
    for key in split_dict:
        all += split_dict[key]

    num_batch = int(length / minimum_size)
    batch_location = list(range(num_batch))
    random.shuffle(batch_location)

    for key in split_dict:
        selected_num_batch = int(num_batch * split_dict[key] / all)
        selected_batch_location = batch_location[-selected_num_batch:]
        del batch_location[-selected_num_batch:]
        selected_location = [range(minimum_size * i, minimum_size * (i + 1)) for i in selected_batch_location]
        selected_location = np.array(selected_location).reshape(-1)
        selected_location.sort()
        result[key] = data.iloc[selected_location]
    return result


def split_data_set_by_date(features_list, dates, minimum_size=128, processes=0):
    if processes <= 0:
        processes = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=processes)

    train_set, validate_set, test_set = [], [], []

    def callback(result):
        train_set.append(result['train'])
        validate_set.append(result['validate'])
        test_set.append(result['test'])

    for features in features_list:
        pool.apply_async(split_data_by_date, (features, dates, minimum_size,), callback=callback)

    pool.close()
    pool.join()
    return train_set, validate_set, test_set


def split_data_by_date(data, dates, minimum_size):
    split_train_validate_date = dates[0]
    split_validate_test_date = dates[1]

    idxs = pd.IndexSlice

    train = data.loc[idxs[:, :split_train_validate_date], idxs[:]]
    round_int = len(train) // minimum_size * minimum_size
    train = train.tail(round_int)

    validate = data.loc[idxs[:, split_train_validate_date:split_validate_test_date], idxs[:]]
    round_int = len(validate) // minimum_size * minimum_size
    validate = validate.tail(round_int)

    test = data.loc[idxs[:, split_validate_test_date:], idxs[:]]
    round_int = len(test) // minimum_size * minimum_size
    test = test.tail(round_int)

    result = {'train': train, 'validate': validate, 'test': test}
    return result


def reform_X_Y(data, timesteps, target_field='label'):
    columns = data.columns.tolist()
    if target_field not in columns:
        return None, None
    x_columns = columns
    x_columns.remove(target_field)
    X, Y0 = data.loc[:, x_columns].values, data.loc[:, target_field].values
    X = X.reshape((-1, timesteps, X.shape[1]))
    Y = np.array([np.array(y) for y in Y0])
    Y = Y.reshape((-1, timesteps, Y.shape[1]))
    return X, Y


# performance_score
def performance_factory(reverse_func, performance_types=['returns']):
    def performance_measures(pct_chg, y):
        result = {}
        y_init = list(map(reverse_func, y))
        predict = pd.Series(index=pct_chg.index, data=y_init)
        predict.name = 'label'
        df = pd.concat([pct_chg, predict], axis=1)
        df['return'] = 0
        epsilon = 0.0001
        long_cond = (abs(df['label'] - 2)) < epsilon
        short_cond = (abs(df['label'])) < epsilon
        df.loc[long_cond, 'return'] = pct_chg.loc[long_cond]/100.0
        df.loc[short_cond, 'return'] = -pct_chg[short_cond]/100.0
        returns = df['return']

        if 'Y' in performance_types:
            result['Y'] = predict
        if 'returns' in performance_types:
            result['returns'] = returns
        if 'cum_returns' in performance_types:
            result['cum_returns'] = empyrical.cum_returns(returns)
        if 'annual_return' in performance_types:
            result['annual_return'] = empyrical.annual_return(returns)
        if 'sharpe_ratio' in performance_types:
            result['sharpe_ratio'] = empyrical.sharpe_ratio(returns)
        return result
    return performance_measures

