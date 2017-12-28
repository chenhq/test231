from functools import partial
from data_prepare import get_data
from feature.construct_feature import ma, label_by_ma_price, construct_features
from index_components import zz500
import matplotlib.pylab as plt
import seaborn as sbn
sbn.set()

if __name__ == '__main__':
    params_list = []
    func_list = []

    ma_params = {
        'ma_list': [1, 2, 3, 5, 8, 13, 21, 34, 55],
        'window': 128,
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

    data_set, reverse_func = get_data(file_name="E:\market_data/cs_market.csv", stks=zz500[:1],
                                      construct_feature_func=construct_feature_func,
                                      split_dates=["2016-01-01", "2017-01-01"])

    train = data_set['train']
    print(train.columns)
    i_columns = ['ma_1', 'ma_2', 'ma_3', 'ma_5', 'ma_8', 'ma_13', 'ma_21', 'ma_34','ma_55']

    train.loc[:, i_columns].plot(figsize=(21, 7))
    plt.show()


