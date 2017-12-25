from functools import partial
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sbn
import _pickle as pickle
from data_prepare import get_data, construct_features1, construct_features2

sbn.set()

if __name__ == '__main__':
    construct_feature_func = construct_features1
    construct_feature_func = partial(construct_features2, ma=5, n_std=0.2, std_window=30, test=True)
    data_set, reverse_func = get_data(file_name="E:\market_data/cs_market.csv",
                                      construct_feature_func=construct_feature_func,
                                      split_dates=["2016-01-01", "2017-01-01"])

    for tag in ['train', 'validate', 'test']:
        data_set[tag]['label2'] = data_set[tag]['label'].map(reverse_func)
        up = data_set[tag][data_set[tag]['label2'] == 2]
        down = data_set[tag][data_set[tag]['label2'] == 0]
        same = data_set[tag][data_set[tag]['label2'] == 1]
        print("up: {}, down: {}, same: {}".format(len(up) / len(data_set[tag]),
                                                  len(down) / len(data_set[tag]),
                                                  len(same) / len(data_set[tag])))
    idx_slice = pd.IndexSlice
    for stk in data_set['train'].index.get_level_values('code').unique():
        train_ohlcv = data_set['train'].loc[idx_slice[stk, :], idx_slice[:]].reset_index().reset_index()
        # validate_ohlcv = data_set['validate'].loc[idx_slice[stk, :], idx_slice[:]].reset_index().reset_index()
        # test_ohlcv = data_set['test'].loc[idx_slice[stk, :], idx_slice[:]].reset_index().reset_index()
        # validate_ohlcv['label2'] = validate_ohlcv['label'].map(reverse_func)
        # test_ohlcv['label2'] = test_ohlcv['label'].map(reverse_func)

        fig, ax = plt.subplots(1, figsize=(21, 7))
        train_ohlcv.plot(x='index', y='close', figsize=(21, 7), ax=ax)
        up = train_ohlcv[train_ohlcv['label2'] == 2]
        if len(up) > 0:
            up.plot.scatter(x='index', y='close', s=10, c='r', figsize=(21, 7), ax=ax)
        down = train_ohlcv[train_ohlcv['label2'] == 0]
        if len(down) > 0:
            down.plot.scatter(x='index', y='close', s=10, c='g', figsize=(21, 7), ax=ax)
        same = train_ohlcv[train_ohlcv['label2'] == 1]
        if len(same) > 0:
            same.plot.scatter(x='index', y='close', s=10, c='b', figsize=(21, 7), ax=ax)
        print("up: {}, down: {}, same: {}".format(len(up)/len(train_ohlcv), len(down)/len(train_ohlcv),
                                                  len(same)/len(train_ohlcv)))
        plt.show()
