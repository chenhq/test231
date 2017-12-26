import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from valid_wave import tag_wave_direction

from data_prepare import split_data_set_by_date
from index_components import zz500
from target.valid_wave.valid_wave_hyperopt import valid_wave_by_multi_processes, get_data

if __name__ == '__main__':
    function = tag_wave_direction
    # best_params = {
    #     'return_per_count_threshold': 0.30000000000000004,
    #     'max_return_threshold': 5.0,
    #     'withdraw_threshold': 4.0,
    #     'std_window': 10.0,
    #     'minimum_period': 10.43056594800402
    # }

    # best_params = {
    #     'minimum_period': 9.389675833299252,
    #     'std_window': 20.0,
    #     'withdraw_threshold': 5.0,
    #     'max_return_threshold': 1.5,
    #     'return_per_count_threshold': 0.5
    # }

    best_params = {
        'std_window': 40,
        'max_return_threshold': 3,
        'return_per_count_threshold': 0.3,
        'withdraw_threshold': 2,
        'minimum_period': 5
    }
    sub_dir = 'relative'

    # function = tag_wave_direction_by_absolute()
    # space = absolute_spaces
    # sub_dir = 'absolute'

    ohlcv_list = get_data("E:\market_data/cs_market.csv", zz500[50:100])
    split_dates = ["2016-01-01", "2017-01-01"]

    log_dir = os.path.join('./logs/valid_wave_hyperopt', sub_dir)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    operation = 'label'
    # operation = 'search'
    tagged_ohlcv_list = valid_wave_by_multi_processes(best_params, ohlcv_list, operation, 'relative')
    train_list, validate_list, test_list = split_data_set_by_date(tagged_ohlcv_list, split_dates, minimum_size=1)
    train_set = pd.concat(train_list, axis=0)
    validate_set = pd.concat(validate_list, axis=0)
    test_set = pd.concat(test_list, axis=0)

    for data in [train_set, validate_set, test_set]:
        up = data[data['direction'] > 0]
        down = data[data['direction'] < 0]
        same = data[np.isnan(data['direction'])]
        print("up: {}, down: {}, same: {}".format(len(up)/len(data), len(down)/len(data), len(same)/len(data)))
        # up: 0.3387733229405174, down: 0.25796917880515097, same: 0.4032574982543317
        # up: 0.2574011700053182, down: 0.2390533593334515, same: 0.5035454706612302
        # up: 0.26273805962708147, down: 0.25675540931299234, same: 0.4805065310599262

    for tagged_ohlcv in tagged_ohlcv_list:
        tagged_ohlcv = tagged_ohlcv.reset_index().reset_index()
        fig, ax = plt.subplots(1, figsize=(21, 7))
        tagged_ohlcv.plot(x='index', y='close', figsize=(21, 7), ax=ax)
        tagged_ohlcv[tagged_ohlcv['direction'] > 0].plot.scatter(x='index', y='close', s=10, c='r', figsize=(21, 7), ax=ax)
        tagged_ohlcv[tagged_ohlcv['direction'] < 0].plot.scatter(x='index', y='close', s=10, c='g', figsize=(21, 7), ax=ax)
        tagged_ohlcv[np.isnan(tagged_ohlcv['direction'])].plot.scatter(x='index', y='close', s=10, c='b', figsize=(21, 7), ax=ax)
        # tagged_ohlcv[tagged_ohlcv['direction'] == 0].plot.scatter(x='index', y='close', s=10, c='b', figsize=(21, 7), ax=ax)
        plt.show()
