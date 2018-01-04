# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from params_select import *
from objective import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial, rand, space_eval
from loss import weighted_categorical_crossentropy3

if __name__ == '__main__':
    params = {
        'ma': 5,
        'std_window': 20,
        'vol_window': 15
    }
    construct_feature_func = partial(construct_features1, params=params, test=False)

    data_set, reverse_func = get_data(file_name="E:\market_data/cs_market.csv", stks=zz500[:50],
                                      construct_feature_func=construct_feature_func,
                                      split_dates=["2016-01-01", "2017-01-01"])
    performance_func = performance_factory(reverse_func,
                                           performance_types=['Y0', 'Y', 'returns', 'cum_returns', 'annual_return',
                                                              'sharpe_ratio'])

    function = "test_weight"
    identity = str(uuid.uuid1())
    namespace = function + '_' + identity

    # loss = 'categorical_crossentropy'
    loss = weighted_categorical_crossentropy3
    objective_func = lstm_objective(data_set, target_field='label', namespace=namespace,
                                    performance_func=performance_func, measure='sharpe_ratio',
                                    loss=loss, include_test_data=True, shuffle_test=False)

    trials = Trials()

    # best = fmin(objective_func, space, algo=tpe.suggest, max_evals=50, trials=trials)
    # print(best)
    params = {
        'activation': 'relu', 'shuffle': False, 'initializer': glorot_uniform(seed=123), 'time_steps': 64, 'units2': 64,
        'is_BN_2': True, 'units1': 128, 'is_BN_1': False, 'lr': 0.0008180455150618453, 'dropout': 0,
        'units3': 256, 'recurrent_dropout': 0.5, 'batch_size': 128, 'epochs': 10, 'is_BN_3': True}

    objective_func(params)

    # params = space_eval(space, best)
    # for i in range(10):
    #     objective_func(params)
