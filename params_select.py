# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from performance import performance_factory
from keras.initializers import glorot_uniform
from objective import objective
from trial import run_a_trial
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial, rand, space_eval
from index_components import sz50, hs300, zz500, zz500_t10
import uuid
import seaborn as snb
import os
from data_prepare import *
snb.set()
from loss import *

try:
    import cPpickle as pickle
except:
    import pickle


lstm_space = {
    'time_steps': hp.choice('time_steps', [32, 64]),
    'batch_size': hp.choice('batch_size', [64, 128]),
    'epochs': hp.choice('epochs', [100, 200, 300, 400, 500]),  # [100, 200, 500, 1000, 1500, 2000]
    'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh', 'linear']),
    # for class
    'activation_last': hp.choice('activation_last', ['softmax']),
    # for regression
    # 'activation_last': hp.choice('activation', [None, 'linear']),
    'shuffle': hp.choice('shuffle', [False, True]),
    'loss_type': hp.choice('loss', ['categorical_crossentropy', 'weighted_categorical_crossentropy']),

    'units1': hp.choice('units1', [64, 128, 256, 512]),
    'units2': hp.choice('units2', [128, 256, 512, 1024]),
    'units3': hp.choice('units3', [64, 128, 256, 512]),

    'is_BN_1': hp.choice('is_BN_1', [False, True]),
    'is_BN_2': hp.choice('is_BN_2', [False, True]),
    'is_BN_3': hp.choice('is_BN_3', [False, True]),

    'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
    'dropout': hp.quniform('dropout', 0.3, 0.5, 0.1),
    'recurrent_dropout': hp.quniform('recurrent_dropout', 0.2, 0.5, 0.1),
    'initializer': hp.choice('initializer', [glorot_uniform(seed=123)]),
}

features_space = {
    'kline': [
        hp.choice('window', [30, 60, 120, 240, 480, 960])
    ],
    'ma': [
        hp.choice('ma_list', [[1, 2, 3, 5, 8, 13, 21, 34, 55]]),
        hp.choice('window', [256]),
        hp.choice('price', ['close'])
    ],
    'label_by_ma_price': [
        hp.choice('label_window', [256]),
        hp.choice('next_ma_window', [3, 5, 10]),
        hp.choice('quantile_list', [[0, 0.1, 0.3, 0.7, 0.9, 1]])
    ]
}

space = {
    'features': features_space,
    'lstm': lstm_space,
    'split_dates': ["2016-01-01", "2017-01-01"]
}


if __name__ == '__main__':
    ohlcv_list = get_data(file_name="~/cs_market.csv", stks=zz500_t10)

    function = "params_select"
    # identity = str(uuid.uuid1())
    identity = 'test02'
    print("identity: {}".format(identity))
    namespace = function + '_' + identity

    log_dir = os.path.join('./', namespace)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, 'stk_ohlcv_list.pkl'), 'wb') as f:
        pickle.dump(ohlcv_list, f)

    # # loss
    # loss = 'categorical_crossentropy'
    # loss = weighted_categorical_crossentropy5
    objective_func = partial(objective, ohlcv_list=ohlcv_list, namespace=namespace)

    trials_file = os.path.join(log_dir, 'trials.pkl')

    while True:
        run_a_trial(trials_file, objective_func, space)





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