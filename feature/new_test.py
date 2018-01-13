from functools import partial
from data_prepare import get_data
from feature.construct_feature import *
from index_components import zz500
import matplotlib.pylab as plt
import seaborn as sbn
from data_prepare import construct_features_for_stocks

sbn.set()

if __name__ == '__main__':
    params_list = []
    func_list = []

    # ma_params = {
    #     'ma_list': [1, 2, 3, 5, 8, 13, 21, 34, 55],
    #     'window': 256,
    #     'price': 'close'
    # }
    # params_list.append(ma_params)
    # func_list.append(feature_ma)

    # label_by_ma_price_params = {
    #     'window': 250,
    #     'next_ma_window': 3,
    #     'quantile_list': [0, 0.1, 0.3, 0.7, 0.9, 1]
    # }
    # params_list.append(label_by_ma_price_params)
    # func_list.append(label_by_ma_price)

    kline2_params = {
        'window': 256,
    }
    params_list.append(kline2_params)
    func_list.append(feature_kline2)

    label_by_multi_ma_params = {
        'window': [3, 5, 10]
    }
    params_list.append(label_by_multi_ma_params)
    func_list.append(label_by_multi_ma)

    construct_feature_func = partial(construct_features, params_list=params_list, func_list=func_list, test=True)

    ohlcv_list = get_data(file_name="~/cs_market.csv", stks=['002277.XSHE'])

    stk_features_list = construct_features_for_stocks(ohlcv_list, construct_feature_func)

    print(len(stk_features_list))
    print(stk_features_list[0].columns)
    # i_columns = ['ma_1', 'ma_2', 'ma_3', 'ma_5', 'ma_8', 'ma_13', 'ma_21', 'ma_34', 'ma_55']
    f = stk_features_list[0]
    f = f.reset_index().reset_index()
    print(f.columns)
    fig, ax = plt.subplots(1, figsize=(21, 7))
    f.loc[:, 'close'].plot(figsize=(21, 7))
    f[f["label"] == -1].plot.scatter(x='index', y='close', s=15, c='green', figsize=(21, 7), ax=ax,
                                     label="down")
    f[f["label"] == 1].plot.scatter(x='index', y='close', s=15, c='red', figsize=(21, 7), ax=ax,
                                    label="up")

    plt.show()
