# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from performance import *
from hyperopt import Trials
from objective import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial, rand, space_eval
import uuid
import seaborn as snb
from loss import *
snb.set()
try:
    import cPpickle as pickle
except:
    import pickle


if __name__ == '__main__':

    default_space = {
        'time_steps': hp.choice('time_steps', [32, 64, 128]),
        'batch_size': hp.choice('batch_size', [64, 128]),
        'epochs': hp.choice('epochs', [100, 200, 300, 400, 500]),  # [100, 200, 500, 1000, 1500, 2000]
        'shuffle': hp.choice('shuffle', [False, True]),

        'units1': hp.choice('units1', [32, 64, 128, 256]),
        'units2': hp.choice('units2', [32, 64, 128, 256]),
        'units3': hp.choice('units3', [32, 64, 128, 256]),

        'is_BN_1': hp.choice('is_BN_1', [False, True]),
        'is_BN_2': hp.choice('is_BN_2', [False, True]),
        'is_BN_3': hp.choice('is_BN_3', [False, True]),

        'lr': hp.uniform('lr', 0.0001, 0.001),
        'dropout': hp.choice('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        'recurrent_dropout': hp.choice('recurrent_dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        'initializer': hp.choice('initializer', [glorot_uniform(seed=123)]),
        # 'min_delta': hp.quniform('min_delta', 0.0002, 0.001, 0.0002),
        # 'patience': hp.quniform('patience', 10, 100, 10),
    }

    data_set, reverse_func = get_data()

    space = default_space

    performance_func = performance_factory(reverse_func,
                                           performance_types=['Y', 'returns', 'cum_returns', 'annual_return',
                                                              'sharpe_ratio'])

    function = "params_select"
    id = str(uuid.uuid1())
    namespace = function + '_' + id
    objective_func = construct_objective2(data_set, target_field='label', namespace=namespace,
                                          performance_func=performance_func, measure='sharpe_ratio',
                                          include_test_data=True, shuffle_test=False, loss=weighted_loss)

    trials = Trials()

    best = fmin(objective_func, space, algo=tpe.suggest, max_evals=5, trials=trials)
    params = space_eval(space, best)
    print(params)
