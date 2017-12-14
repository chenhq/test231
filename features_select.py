# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from performance import *
from objective import *
import seaborn as snb
snb.set()
try:
    import cPpickle as pickle
except:
    import pickle


if __name__ == '__main__':
    data_set, reverse_func = get_data()

    performance_func = performance_factory(reverse_func,
                                           performance_types=['Y', 'returns', 'cum_returns', 'annual_return',
                                                              'sharpe_ratio'])

    function = "features_select"
    id = str(uuid.uuid1())
    namespace = function + '_' + id
    objective_func = construct_objective2(data_set, target_field='label', namespace=namespace,
                                          performance_func=performance_func, measure='sharpe_ratio',
                                          include_test_data=True, shuffle_test=True)


    params = {'batch_size': 64,
            'dropout': 0.5,
            'epochs': 10,
            'initializer': glorot_uniform(seed=123),
            'is_BN_1': True,
            'is_BN_2': True,
            'is_BN_3': True,
            'lr': 0.0009018180968756607,
            'min_delta': 0.005,
            'patience': 5.0,
            'recurrent_dropout': 0.1,
            'shuffle': False,
            'time_steps': 64,
            'units1': 256,
            'units2': 256,
            'units3': 32}

    objective_func(params)







