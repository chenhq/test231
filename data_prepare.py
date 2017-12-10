import numpy as np
import pandas as pd
from talib.abstract import *
import random
import empyrical
import multiprocessing


def construct_features(ohlcv, construct_features_func):
    return construct_features_func(ohlcv)


def construct_features_for_stocks(ohlcv_dict, construct_features_func, processes=0):
    stock_features_dict = {}

    if processes <= 0:
        processes = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=processes)

    def callback(result):
        stk, features = result
        stock_features_dict[stk] = features

    def work(stk, ohlcv):
        stock_features = construct_features_func(ohlcv)
        return stk, stock_features

    for stk in ohlcv_dict:
        ohlcv = ohlcv_dict[stk]
        pool.apply_async(work, (stk, ohlcv, ), callback=callback)

    pool.close()
    pool.join()
    return stock_features_dict


def construct_features1(ohlcv):
    data = pd.DataFrame(index=ohlcv.index)
    data['open_close'] = (ohlcv['open'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['high_close'] = (ohlcv['high'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['low_close'] = (ohlcv['low'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['pct_chg'] = ((ohlcv['close'] / ohlcv['close'].shift(1)) - 1) * 100
    data = data.fillna(0)

    next_ma5 = SMA(ohlcv, timeperiod=5).shift(-5)
    data['label'] = 1
    data.loc[data['close'] < next_ma5, 'label'] = 0
    data.loc[data['close'] > next_ma5, 'label'] = 2

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


def data_set_to_categorical(data_set, column, func, processes=0):
    if processes <= 0:
        processes = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=processes)

    new_data_set = {}

    reverse_func = None

    def callback(result):
        key, new_data = result
        new_data_set[key] = new_data

    def work(key, data):
        new_data, reverse_func = to_categorical(data, column, func)
        return key, new_data

    for key in data_set:
        data = data_set[key]
        pool.apply_async(work, (key, data, ), callback=callback)

    pool.close()
    pool.join()
    return new_data_set, reverse_func


def data_split_by_idx(data, idxes):
    pass


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


def split_data_set_by_date(features_dict, dates, minimum_size=128, processes=0):
    if processes <= 0:
        processes = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=processes)

    train_set, validate_set, test_set = {}, {}, {}

    def callback(result):
        stk, data_set = result
        train_set[stk] = data_set['train']
        validate_set[stk] = data_set['validate']
        test_set[stk] = data_set['test']

    def work(stk, features):
        data_set = split_data_by_date(features, dates, minimum_size=minimum_size)
        return stk, data_set

    for stk in features_dict:
        features = features_dict[stk]
        pool.apply_async(work, (stk, features, ), callback=callback)

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


def reform_X_Y(data, batch_size, timesteps, target_field='label'):
    size = len(data)
    if size % (int(batch_size) * int(timesteps)) != 0:
        print("data size not match, size: {0}, batch_size: {1}, timesteps: {2}".format(size, batch_size, timesteps))
        return None, None

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


def flatten_stock_features(stock_features_set):
    result = pd.DataFrame()
    for stk in stock_features_set:
        stock_features = stock_features_set[stk]
        result = pd.concat([result, stock_features.reset_index()], axis=0)

    result = result.set_index(['code', 'date']).sort_index()
    return result




def split_data_set(data, minimum_size=128, train_ratio=3, test_ratio=1, validate_ratio=1):
    result = split_data_by_sample(data, {"train": train_ratio, "validate": validate_ratio, "test": test_ratio},
                                  minimum_size)
    train = result['train']
    validate = result['validate']
    test = result['test']

    return train, validate, test


# performance_score
def performance_factory(reverse_func, performace_types=['returns']):
    def performance_measures(pct_chg, y):
        result = {}
        y_init = list(map(reverse_func, y))
        predict = pd.Series(index=pct_chg.index, data=y_init)
        predict.name = 'label'
        df = pd.concat([pct_chg, predict], axis=1)
        df['return'] = 0
        epsilon = 0.0001
        df.loc[(abs(df['label'] - 2)) < epsilon, 'return'] = pct_chg/100.0
        df.loc[(abs(df['label'])) < epsilon, 'return'] = -pct_chg/100.0
        returns = df['return']

        if 'returns' in performace_types:
            result['returns'] = returns
        if 'cum_returns' in performace_types:
            result['cum_returns'] = empyrical.cum_returns(returns)
        if 'annual_return' in performace_types:
            result['annual_return'] = empyrical.annual_return(returns)
        if 'sharpe_ratio' in performace_types:
            result['sharpe_ratio'] = empyrical.sharpe_ratio(returns)
        return result
    return performance_measures