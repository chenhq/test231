from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial, rand, space_eval
import pandas as pd
import multiprocessing
from valid_wave import *
import uuid
import empyrical
import os
from index_components import sz50, hs300, zz500
import matplotlib.pylab as plt

try:
    import _pickle as pickle
except:
    import pickle
from functools import partial

absolute_spaces = {
    'max_return_threshold':
        hp.quniform('max_return_threshold', 0.02, 0.2, 0.01),
    'return_per_count_threshold':
        hp.quniform('return_per_count_threshold', 0.001, 0.02, 0.001),
    'withdraw_threshold':
        hp.quniform('withdraw_threshold', 0.005, 0.1, 0.005)
}

relative_spaces = {
    "window":
        hp.quniform('window', 4, 100, 2),
    'max_return_threshold':
        hp.quniform('max_return_threshold', 2, 30, 1),
    'return_per_count_threshold':
        hp.quniform('return_per_count_threshold', 0.1, 5, 0.1),
    'withdraw_threshold':
        hp.quniform('withdraw_threshold', 0.5, 20, 0.5)
}


def get_data():
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
    return stk_ohlcv_list


def validate_wave_by_multi_processes(params, valide_wave_func, ohlcv_list, processes=0):
    stk_result_list = []

    if processes <= 0:
        processes = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=processes)

    def callback(result):
        stk_result_list.append(result)

    def print_error(err):
        print(err)

    for ohlcv in ohlcv_list:
        if valide_wave_func == tag_wave_direction_by_absolute:
            pool.apply_async(valide_wave_func, args=(
                ohlcv, params['max_return_threshold'], params['return_per_count_threshold'],
                params['withdraw_threshold'],),
                             callback=callback, error_callback=print_error)
        if valide_wave_func == tag_wave_direction_by_relative:
            pool.apply_async(valide_wave_func, args=(
                ohlcv, params['window'], params['max_return_threshold'], params['return_per_count_threshold'],
                params['withdraw_threshold'],),
                             callback=callback, error_callback=print_error)

    pool.close()
    pool.join()
    return stk_result_list


def objective(params, function, ohlcv_list, log_dir):
    print(params)
    identity = str(uuid.uuid1())
    result_list = validate_wave_by_multi_processes(params, function, ohlcv_list)
    returns_list = []
    for result in result_list:
        stk_returns = result['pct_chg'] * result['direction']
        stk_returns = stk_returns.fillna(0)
        (stk_returns + 1).cumprod().plot(figsize=(21, 7))
        plt.show()
        returns_list.append(stk_returns)
    returns = pd.concat(returns_list, axis=0)

    annual_return = empyrical.annual_return(returns)
    sharpe_ratio = empyrical.sharpe_ratio(returns)

    data = {'id': identity, 'returns': returns,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio}

    with open(os.path.join(log_dir, identity + '.pkl'), 'wb') as f:
        pickle.dump(data, f)

    print('id: %s, annual_return: %s, sharpe_ratio: %s' % (identity, annual_return, sharpe_ratio))
    return {'loss': -sharpe_ratio, 'status': STATUS_OK}


if __name__ == '__main__':
    function = tag_wave_direction_by_relative
    space = relative_spaces
    sub_dir = 'relative'

    # function = tag_wave_direction_by_absolute()
    # space = absolute_spaces
    # sub_dir = 'absolute'

    ohlcv_list = get_data()
    log_dir = os.path.join('./valid_wave_hyperopt', sub_dir)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    hyperopt_objective = partial(objective, function=function, ohlcv_list=ohlcv_list, log_dir=log_dir)
    # trials = Trials()
    # best = fmin(hyperopt_objective, space, algo=tpe.suggest, max_evals=60, trials=trials)
    # params = space_eval(space, best)
    # print(params)

    params = {
        'window': 30,
        'max_return_threshold': 5,
        'return_per_count_threshold': 0.02,
        'withdraw_threshold': 3}

    hyperopt_objective(params)
