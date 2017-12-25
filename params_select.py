# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from performance import performance_factory
from hyperopt import Trials
from keras.initializers import glorot_uniform
from objective import construct_objective
from data_prepare import get_data, construct_features1, construct_features2, construct_features3
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial, rand, space_eval
import uuid
import seaborn as snb
snb.set()
from loss import weighted_categorical_crossentropy
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
        'shuffle': hp.choice('shuffle', [False, True]),

        'units1': hp.choice('units1', [32, 64, 128, 256, 512, 1024]),
        'units2': hp.choice('units2', [32, 64, 128, 256, 512, 1024]),
        'units3': hp.choice('units3', [32, 64, 128, 256, 512, 1024]),

        'is_BN_1': hp.choice('is_BN_1', [False, True]),
        'is_BN_2': hp.choice('is_BN_2', [False, True]),
        'is_BN_3': hp.choice('is_BN_3', [False, True]),

        'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
        'dropout': hp.quniform('dropout', 0, 0.5, 0.1),
        'recurrent_dropout': hp.quniform('recurrent_dropout', 0, 0.5, 0.1),
        'initializer': hp.choice('initializer', [glorot_uniform(seed=123)]),
        # 'min_delta': hp.quniform('min_delta', 0.0002, 0.001, 0.0002),
        # 'patience': hp.quniform('patience', 10, 100, 10),
    }

    # features
    params = {
        'ma': 5,
        'std_window': 20,
        'vol_window': 15
    }
    construct_feature_func = partial(construct_features1, params=params, test=False)

    # params = {
    #     'ma': 5,
    #     'n_std': 0.3,
    #     'std_window': 30,
    #     'vol_window': 15
    # }
    # construct_feature_func = partial(construct_features2, params=params, test=False)
    #
    # params = {
    #     'std_window': 40,
    #     'vol_window': 15,
    #     'max_return_threshold': 3,
    #     'return_per_count_threshold': 0.3,
    #     'withdraw_threshold': 2,
    #     'minimum_period': 5
    # }
    # construct_feature_func = partial(construct_feature_func3, params=params, test=False)

    data_set, reverse_func = get_data(file_name="~/cs_market.csv",
                                      construct_feature_func=construct_feature_func,
                                      split_dates=["2016-01-01", "2017-01-01"])

    space = default_space

    performance_func = performance_factory(reverse_func,
                                           performance_types=['Y', 'returns', 'cum_returns', 'annual_return',
                                                              'sharpe_ratio'])

    function = "params_select"
    identity = str(uuid.uuid1())
    namespace = function + '_' + identity

    # loss
    loss = 'categorical_crossentropy'
    # loss = weighted_categorical_crossentropy
    objective_func = construct_objective(data_set, target_field='label', namespace=namespace,
                                         performance_func=performance_func, measure='sharpe_ratio',
                                         include_test_data=True, shuffle_test=False,
                                         loss=loss)

    trials = Trials()

    best = fmin(objective_func, space, algo=tpe.suggest, max_evals=50, trials=trials)
    best_params = space_eval(space, best)
    print("best_params: {}".format(best_params))
