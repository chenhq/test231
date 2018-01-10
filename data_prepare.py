import multiprocessing
import random
import _pickle as pickle
import numpy as np
import pandas as pd


def construct_features_for_stocks(ohlcv_list, construct_features_func, processes=0):
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
    class_list = data[column].unique().tolist()
    class_list.sort()
    num_class = len(class_list)
    data[column] = data[column].copy().map(func(num_class, class_list))

    def reverse_func(array):
        return class_list[np.argmax(array)]

    return data, reverse_func


def categorical_factory(class_list):
    def to_categorical(x):
        cls = []
        for i in range(len(class_list)):
            if class_list[i] == x:
                cls.append(1.0)
            else:
                cls.append(0.0)
        return np.array(cls)

    def reverse_categorical(array):
        return class_list[np.argmax(array)]

    return to_categorical, reverse_categorical


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


def split_data_set_by_date(features_list, dates, minimum_size=64, processes=0):
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
    train = pd.concat(train_set, axis=0)
    validate = pd.concat(validate_set, axis=0)
    test = pd.concat(test_set, axis=0)
    data_set = {'train': train, 'validate': validate, 'test': test}
    return data_set


def split_data_by_date(data, dates, minimum_size=1):
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


# "../data/cs_market.csv"
# "~/cs_market.csv"
# "E:\market_data/cs_market.csv"
def get_data(file_name, stks):
    market = pd.read_csv(file_name, parse_dates=["date"], dtype={"code": str})
    all_ohlcv = market.drop(["Unnamed: 0", "total_turnover", "limit_up", "limit_down"], axis=1)
    all_ohlcv = all_ohlcv.set_index(['code', 'date']).sort_index()
    idx_slice = pd.IndexSlice
    stk_ohlcv_list = []
    for stk in all_ohlcv.index.get_level_values('code').unique():
        if stk in stks:
            stk_ohlcv = all_ohlcv.loc[idx_slice[stk, :], idx_slice[:]]
            stk_ohlcv_list.append(stk_ohlcv)
    return stk_ohlcv_list


def get_pickle_data(pickle_file, stks):
    idx_slice = pd.IndexSlice
    stk_ohlcv_list = []
    all_ohlcv = pickle.load(open(pickle_file, 'rb'))
    for stk in all_ohlcv.index.get_level_values('code').unique():
        if len(stks) == 0 or stk in stks:
            stk_ohlcv = all_ohlcv.loc[idx_slice[stk, :], idx_slice[:]]
            stk_ohlcv_list.append(stk_ohlcv)
    return stk_ohlcv_list
