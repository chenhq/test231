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
    import _pickle as pickle
except:
    import pickle


lstm_space = {
    'time_steps': hp.choice('time_steps', [64]),
    'batch_size': hp.choice('batch_size', [64]),
    'epochs': hp.choice('epochs', [100, 200, 300, 400, 500, 800]),  # [100, 200, 500, 1000, 1500, 2000]

    # for class
    'activation_last': hp.choice('activation_last', ['softmax']),
    # for regression
    # 'activation_last': hp.choice('activation', [None, 'linear']),

    #
    'shuffle': hp.choice('shuffle', [False]),
    'loss_type': hp.choice('loss', ['categorical_crossentropy']), #, 'weighted_categorical_crossentropy']),

    'layer1': {
        'units': hp.choice('layer1_units', [32, 64, 128, 256]),
        # 'relu', 'sigmoid', 'tanh', 'linear'
        'activation': hp.choice('layer1_activation', ['tanh']),
        'is_BN': hp.choice('layer1_is_BN', [False, True]),
    },
    'layer2': {
        'units': hp.choice('layer2_units', [32, 64, 128, 256]),
        # 'relu', 'sigmoid', 'tanh', 'linear'
        'activation': hp.choice('layer2_activation', ['tanh']),
        'is_BN': hp.choice('layer2_is_BN', [False, True]),
    },
    'layer3': {
        'units': hp.choice('layer3_units', [32, 64, 128, 256]),
        # 'relu', 'sigmoid', 'tanh', 'linear'
        # Loss turns into 'nan'
        # As far as I know, it's the combination of relu and softmax that causes numerical troubles,
        # as relu can produce large positive values corresponding to very small probabilities.
        # If you change your model to use, say, tanh instead of relu for the last dense layer,
        # the problem will go away.
        'activation': hp.choice('layer3_activation', ['tanh']),
        'is_BN': hp.choice('layer3_is_BN', [False, True]),
    },

    'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
    'dropout': hp.quniform('dropout', 0.3, 0.31, 0.1),
    'recurrent_dropout': hp.quniform('recurrent_dropout', 0.3, 0.31, 0.1),
    'kernel_initializer': hp.choice('kernel_initializer', [glorot_uniform(seed=123)]),
    'bias_initializer': hp.choice('bias_initializer', [glorot_uniform(seed=456)]),
}

features_space = {
    'kline': {
        'window': hp.choice('kline_window', [240])
    },
    'ma': {
        'ma_list': hp.choice('ma_list', [[1, 2, 3, 5, 8, 13, 21, 34, 55, 60, 120, 240, 480]]),
        'window': hp.choice('ma_window', [240]),
        'price': hp.choice('price', ['close'])
    },
    'label_by_ma_price': {
        'window': hp.choice('label_window', [120]),
        'next_ma_window': hp.choice('next_ma_window', [2, 3]),
        'quantile_list': hp.choice('quantile_list', [# [0, 0.1, 0.3, 0.7, 0.9, 1],
                                                     # [0, 0.2, 0.4, 0.6, 0.8, 1],
                                                     # [0, 0.15, 0.3, 0.7, 0.85, 1],
                                                     # [0, 0.15, 0.35, 0.65, 0.85, 1],
                                                     # [0, 0.3, 0.7, 1],
                                                     [0, 0.33, 0.66, 1],
                                                     # [0, 0.2, 0.8, 1],
                                                     # [0, 0.4, 0.6, 1],
                                                     # [0, 0.5, 1],
                                                     # [0, 0.45, 1],
                                                     # [0, 0.55, 1]
                                                    ])
    }
}

space = {
    'features': features_space,
    'lstm': lstm_space,
    'split_dates': ["2016-01-01", "2017-01-01"]
}


if __name__ == '__main__':
    file_name = '../data/cs_market.csv'
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
