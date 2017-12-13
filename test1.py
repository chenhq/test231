# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# %load test1.py
from hyperopt import Trials

from index_components import *
from params_select import *

if __name__ == '__main__':
    # market = pd.read_csv("../data/cs_market.csv", parse_dates=["date"], dtype={"code": str})
    # market = pd.read_csv("~/cs_market.csv", parse_dates=["date"], dtype={"code": str})
    market = pd.read_csv("E:\market_data/cs_market.csv", parse_dates=["date"], dtype={"code": str})
    all_ohlcv = market.drop(["Unnamed: 0", "total_turnover", "limit_up", "limit_down"], axis=1)

    all_ohlcv = all_ohlcv.set_index(['code', 'date']).sort_index()

    idx_slice = pd.IndexSlice

    stk_ohlcv_list = []
    for stk in all_ohlcv.index.get_level_values('code').unique():
        if stk in sz50:
            stk_ohlcv = all_ohlcv.loc[idx_slice[stk, :], idx_slice[:]]
            stk_ohlcv_list.append(stk_ohlcv)

    stk_features_list = construct_features_for_stocks(stk_ohlcv_list, construct_features1)

    flatten_stk_features_list, reverse_func = to_categorical(pd.concat(stk_features_list, axis=0), 'label',
                                                             categorical_func_factory)

    new_stk_features_list = []
    for stk in flatten_stk_features_list.index.get_level_values('code').unique():
        new_stk_features = flatten_stk_features_list.loc[idx_slice[stk, :], idx_slice[:]]
        new_stk_features_list.append(new_stk_features)

    split_dates = ["2016-01-01", "2017-01-01"]

    train_set, validate_set, test_set = split_data_set_by_date(new_stk_features_list, split_dates, minimum_size=128)

    train = pd.concat(train_set, axis=0)
    validate = pd.concat(validate_set, axis=0)
    test = pd.concat(test_set, axis=0)

    data_set = {'train': train, 'validate': validate, 'test': test}

    space = default_space

    performance_measure = performance_factory(reverse_func,
                                              performance_types=['Y', 'returns', 'cum_returns', 'annual_return',
                                                                 'sharpe_ratio'])

    objective_func = construct_objective2(data_set, "logs", performance_measure, 'sharpe_ratio', include_test_data=True)

    trials = Trials()

    # best = fmin(objective_func, space, algo=tpe.suggest, max_evals=50, trials=trials)
    # print(best)

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

    # params = space_eval(space, best)
    # for i in range(10):
    #     objective_func(params)
