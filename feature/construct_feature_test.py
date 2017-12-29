import numpy as np
from functools import partial
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sbn
import _pickle as pickle
from data_prepare import get_data
from feature.construct_feature import *
from index_components import sz50, zz500, hs300
import matplotlib.cm as cm

sbn.set()

if __name__ == '__main__':
    # params = {
    #     'ma': 5,
    #     'std_window': 20,
    #     'vol_window': 15
    # }
    # construct_feature_func = partial(construct_features1, params=params, test=True)
    #
    # params = {
    #     'ma': 5,
    #     'n_std': 0.3,
    #     'std_window': 30,
    #     'vol_window': 15
    # }
    # construct_feature_func = partial(construct_features2, params=params, test=True)

    # params = {
    #     'std_window': 40,
    #     'vol_window': 15,
    #     'max_return_threshold': 3,
    #     'return_per_count_threshold': 0.3,
    #     'withdraw_threshold': 2,
    #     'minimum_period': 5
    # }
    # construct_feature_func = partial(construct_features3, params=params, test=True)

    # params = {
    #     'window': 250,
    #     'next_ma_window': 3,
    #     'quantile_list': [0, 0.1, 0.3, 0.7, 0.9, 1]
    # }
    # construct_feature_func = partial(label_by_ma_price, params=params, test=True)


    params_list = []
    func_list = []

    ma_params = {
        'ma_list': [1, 2, 3, 5, 8, 13, 21, 34, 55],
        'window': 256,
        'price': 'close'
    }
    params_list.append(ma_params)
    func_list.append(ma)

    label_by_ma_price_params = {
        'window': 250,
        'next_ma_window': 3,
        'quantile_list': [0, 0.1, 0.3, 0.7, 0.9, 1]
    }
    params_list.append(label_by_ma_price_params)
    func_list.append(label_by_ma_price)

    construct_feature_func = partial(construct_features, params_list=params_list, func_list=func_list, test=True)

    data_set, reverse_func = get_data(file_name="E:\market_data/cs_market.csv", stks=zz500[200:205],
                                      construct_feature_func=construct_feature_func,
                                      split_dates=["2016-01-01", "2017-01-01"])

    for tag in ['train', 'validate', 'test']:
        data_set[tag]['label2'] = data_set[tag]['label'].map(reverse_func)
        labels = data_set[tag]['label2'].unique().tolist()
        labels.sort()
        print(tag)
        for label in labels:
            selected = data_set[tag][data_set[tag]['label2'] == label]
            print("{}: {}".format(label, len(selected)/len(data_set[tag])))

    idx_slice = pd.IndexSlice
    stks = data_set['train'].index.get_level_values('code').unique().tolist()
    stks.sort()
    for stk in stks:
        print(stk)
        train_ohlcv = data_set['train'].loc[idx_slice[stk, :], idx_slice[:]].reset_index().reset_index()
        # validate_ohlcv = data_set['validate'].loc[idx_slice[stk, :], idx_slice[:]].reset_index().reset_index()
        # test_ohlcv = data_set['test'].loc[idx_slice[stk, :], idx_slice[:]].reset_index().reset_index()
        # validate_ohlcv['label2'] = validate_ohlcv['label'].map(reverse_func)
        # test_ohlcv['label2'] = test_ohlcv['label'].map(reverse_func)

        fig, ax = plt.subplots(1, figsize=(21, 7))
        i_columns = ['ma_1', 'ma_2', 'ma_3', 'ma_5', 'ma_8', 'ma_13', 'ma_21', 'ma_34', 'ma_55']
        train_ohlcv.loc[:, i_columns].plot(figsize=(21, 7))
        train_ohlcv.plot(x='index', y='close', figsize=(21, 7), ax=ax)

        labels = train_ohlcv['label2'].unique().tolist()
        labels.sort()
        colors = cm.rainbow(np.linspace(0, 1, len(labels)))
        for i in range(len(labels)):
            label = labels[i]
            c = colors[i]
            selected = train_ohlcv[train_ohlcv['label2'] == label]
            print("{}: {}".format(label, len(selected) / len(train_ohlcv)))
            if len(selected) > 0:
                selected.plot.scatter(x='index', y='close', s=10, c=c, figsize=(21, 7), ax=ax, label=label)

        # if len(up) > 0:
        #     up.plot.scatter(x='index', y='close', s=10, c='r', figsize=(21, 7), ax=ax)
        # down = train_ohlcv[train_ohlcv['label2'] == 0]
        # if len(down) > 0:
        #     down.plot.scatter(x='index', y='close', s=10, c='g', figsize=(21, 7), ax=ax)
        # same = train_ohlcv[train_ohlcv['label2'] == 1]
        # if len(same) > 0:
        #     same.plot.scatter(x='index', y='close', s=10, c='b', figsize=(21, 7), ax=ax)
        # print("up: {}, down: {}, same: {}".format(len(up)/len(train_ohlcv), len(down)/len(train_ohlcv),
        #                                           len(same)/len(train_ohlcv)))
        plt.show()
