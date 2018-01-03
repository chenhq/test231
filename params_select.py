# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from performance import performance_factory
from keras.initializers import glorot_uniform
from objective import construct_objective
from data_prepare import get_data
from feature.construct_feature import *
from hyperopt import hp
from index_components import sz50, hs300, zz500
from trial import run_a_trial
import uuid
import seaborn as snb
import os
snb.set()
from loss import *
from functools import partial

try:
    import cPpickle as pickle
except:
    import pickle


if __name__ == '__main__':

    default_space = {
        'time_steps': hp.choice('time_steps', [32, 64]),
        'batch_size': hp.choice('batch_size', [64, 128]),
        'epochs': hp.choice('epochs', [100, 200, 300, 400, 500]),  # [100, 200, 500, 1000, 1500, 2000]
        'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh', 'linear']),
        # for class
        'activation_last': hp.choice('activation_last', ['softmax']),
        # for regression
        # 'activation_last': hp.choice('activation', [None, 'linear']),
        'shuffle': hp.choice('shuffle', [False, True]),

        'units1': hp.choice('units1', [128, 256, 512]),
        'units2': hp.choice('units2', [256, 512, 1024]),
        'units3': hp.choice('units3', [128, 256, 512]),

        'is_BN_1': hp.choice('is_BN_1', [False, True]),
        'is_BN_2': hp.choice('is_BN_2', [False, True]),
        'is_BN_3': hp.choice('is_BN_3', [False, True]),

        'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
        'dropout': hp.quniform('dropout', 0.3, 0.5, 0.1),
        'recurrent_dropout': hp.quniform('recurrent_dropout', 0.2, 0.5, 0.1),
        'initializer': hp.choice('initializer', [glorot_uniform(seed=123)]),
    }

    # features
    # params = {
    #     'ma': 5,
    #     'std_window': 20,
    #     'vol_window': 15
    # }
    # construct_feature_func = partial(construct_features1, params=params, test=False)

    # params = {
    #     'ma': 5,
    #     'n_std': 0.3,
    #     'std_window': 30,
    #     'vol_window': 15
    # }
    # construct_feature_func = partial(construct_features2, params=params, test=False)

    # params = {
    #     'std_window': 40,
    #     'vol_window': 15,
    #     'max_return_threshold': 3,
    #     'return_per_count_threshold': 0.3,
    #     'withdraw_threshold': 2,
    #     'minimum_period': 5
    # }
    # construct_feature_func = partial(construct_features3, params=params, test=False)

    params_list = []
    func_list = []

    kline_params = {
        'window': 60,
    }
    params_list.append(kline_params)
    func_list.append(features_kline)

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

    construct_feature_func = partial(construct_features, params_list=params_list, func_list=func_list, test=False)

    stks = zz500[100:110]
    print('stks: {}'.format(stks))
    data_set, reverse_func = get_data(file_name="E:\market_data/cs_market.csv", stks=stks,
                                      construct_feature_func=construct_feature_func,
                                      split_dates=["2016-01-01", "2017-01-01"])

    space = default_space

    performance_func = performance_factory(reverse_func,
                                           performance_types=['Y0', 'Y', 'returns', 'cum_returns', 'annual_return',
                                                              'sharpe_ratio'],
                                           mid_type=2, epsilon=0.5)

    function = "params_select"
    # identity = str(uuid.uuid1())
    identity = 'sssssss'
    print("identity: {}".format(identity))
    namespace = function + '_' + identity

    log_dir = os.path.join('./', namespace)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, 'data.pkl'), 'wb') as f:
        pickle.dump(data_set, f)

    # loss
    # loss = 'categorical_crossentropy'
    loss = weighted_categorical_crossentropy5
    objective_func = construct_objective(data_set, target_field='label', namespace=namespace,
                                         performance_func=performance_func, measure='annual_return',
                                         include_test_data=True, shuffle_test=False,
                                         loss=loss)

    trials_file = os.path.join(log_dir, 'trials.pkl')

    while True:
        run_a_trial(trials_file, objective_func, space)


