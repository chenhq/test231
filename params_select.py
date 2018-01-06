# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import os

import seaborn as snb
from hyperopt import hp
from keras.initializers import glorot_uniform

from data_prepare import *
from index_components import zz500_t10
from objective import objective
from trial import run_a_trial

snb.set()
from loss import *

try:
    import cPpickle as pickle
except:
    import pickle


lstm_space = {
    'time_steps': hp.choice('time_steps', [64]),
    'batch_size': hp.choice('batch_size', [64, 128, 256]),
    'epochs': hp.choice('epochs', [100, 200, 300, 400, 500, 800]),  # [100, 200, 500, 1000, 1500, 2000]
    # 'relu', 'sigmoid', 'tanh', 'linear'
    'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh']),
    # for class
    'activation_last': hp.choice('activation_last', ['softmax']),
    # for regression
    # 'activation_last': hp.choice('activation', [None, 'linear']),
    'shuffle': hp.choice('shuffle', [False, True]),
    'loss_type': hp.choice('loss', ['categorical_crossentropy']), #, 'weighted_categorical_crossentropy']),

    'units1': hp.choice('units1', [32, 64, 128, 256]),
    'units2': hp.choice('units2', [32, 64, 128, 256]),
    'units3': hp.choice('units3', [32, 64, 128, 256]),

    'is_BN_1': hp.choice('is_BN_1', [False, True]),
    'is_BN_2': hp.choice('is_BN_2', [False, True]),
    'is_BN_3': hp.choice('is_BN_3', [False, True]),

    'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
    'dropout': hp.quniform('dropout', 0.2, 0.5, 0.1),
    'recurrent_dropout': hp.quniform('recurrent_dropout', 0.2, 0.5, 0.1),
    'initializer': hp.choice('initializer', [glorot_uniform(seed=123)]),
}

features_space = {
    'kline': {
        'window': hp.choice('kline_window', [60, 120, 240, 480, 960])
    },
    'ma': {
        'ma_list': hp.choice('ma_list', [[1, 2, 3, 5, 8, 13, 21, 34, 55, 60, 120, 240, 480]]),
        'window': hp.choice('ma_window', [30, 60, 120, 240, 480, 960]),
        'price': hp.choice('price', ['close'])
    },
    'label_by_ma_price': {
        'window': hp.choice('label_window', [60, 120, 240, 480, 960]),
        'next_ma_window': hp.choice('next_ma_window', [2, 3, 5, 8, 13, 21, 34, 55]),
        'quantile_list': hp.choice('quantile_list', [[0, 0.1, 0.3, 0.7, 0.9, 1],
                                                     [0, 0.2, 0.4, 0.6, 0.8, 1],
                                                     [0, 0.15, 0.3, 0.7, 0.85, 1],
                                                     [0, 0.15, 0.35, 0.65, 0.85, 1],
                                                     [0, 0.3, 0.7, 1],
                                                     [0, 0.33, 0.66, 1],
                                                     [0, 0.2, 0.8, 1],
                                                     [0, 0.4, 0.6, 1],
                                                     [0, 0.5, 1],
                                                     [0, 0.45, 1],
                                                     [0, 0.55, 1]])
    }
}

space = {
    'features': features_space,
    'lstm': lstm_space,
    'split_dates': ["2016-01-01", "2017-01-01"]
}


if __name__ == '__main__':
    file_name = 'E:\market_data/cs_market.csv'
    ohlcv_list = get_data(file_name=file_name, stks=zz500_t10)

    function = "params_select"
    # identity = str(uuid.uuid1())
    identity = 'test03'
    print("identity: {}".format(identity))
    namespace = function + '_' + identity

    namespace = os.path.join('./', namespace)
    if not os.path.exists(namespace):
        os.makedirs(namespace)

    with open(os.path.join(namespace, 'ohlcv_list.pkl'), 'wb') as f:
        pickle.dump(ohlcv_list, f)

    # # loss
    # loss = 'categorical_crossentropy'
    # loss = weighted_categorical_crossentropy5
    objective_func = partial(objective, ohlcv_list=ohlcv_list, namespace=namespace)

    trials_file = os.path.join(namespace, 'trials.pkl')

    while True:
        run_a_trial(trials_file, objective_func, space)
