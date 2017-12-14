# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from params_select import *
from objective import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial, rand, space_eval
from loss import weighted_loss

if __name__ == '__main__':
    data_set, reverse_func = get_data()
    performance_func = performance_factory(reverse_func,
                                           performance_types=['Y', 'returns', 'cum_returns', 'annual_return',
                                                              'sharpe_ratio'])

    function = "test_weight"
    id = str(uuid.uuid1())
    namespace = function + '_' + id
    objective_func = construct_objective2(data_set, target_field='label', namespace=namespace,
                                          performance_func=performance_func, measure='sharpe_ratio',
                                          loss=weighted_loss, include_test_data=True, shuffle_test=False)

    trials = Trials()

    # best = fmin(objective_func, space, algo=tpe.suggest, max_evals=50, trials=trials)
    # print(best)
    params = {
        'shuffle': False, 'initializer': glorot_uniform(seed=123), 'time_steps': 128, 'units2': 64,
        'is_BN_2': True, 'units1': 128, 'is_BN_1': False, 'lr': 0.0008180455150618453, 'dropout': 0,
        'units3': 256, 'recurrent_dropout': 0.5, 'batch_size': 128, 'epochs': 100, 'is_BN_3': True}

    objective_func(params)

    # params = space_eval(space, best)
    # for i in range(10):
    #     objective_func(params)
