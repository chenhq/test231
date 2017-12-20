import multiprocessing
import os
import uuid

import empyrical
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from data_prepare import  split_data_set_by_date
from index_components import zz500
from valid_wave import *

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
        hp.quniform('withdraw_threshold', 0.005, 0.1, 0.005),
    'minimum_period':
        hp.uniform('minimum_period', 5, 20)
}

relative_spaces = {
    'std_window':
        hp.quniform('window', 10, 60, 5),
    'max_return_threshold':
        hp.quniform('max_return_threshold', 1.0, 8.0, 0.5),
    'return_per_count_threshold':
        hp.quniform('return_per_count_threshold', 0.1, 1.0, 0.1),
    'withdraw_threshold':
        hp.quniform('withdraw_threshold', 0.5, 5, 0.5),
    'minimum_period':
        hp.uniform('minimum_period', 5, 20)
}


def get_data(stk_list):
    # market = pd.read_csv("../data/cs_market.csv", parse_dates=["date"], dtype={"code": str})
    # market = pd.read_csv("~/cs_market.csv", parse_dates=["date"], dtype={"code": str})
    market = pd.read_csv("E:\market_data/cs_market.csv", parse_dates=["date"], dtype={"code": str})
    all_ohlcv = market.drop(["Unnamed: 0", "total_turnover", "limit_up", "limit_down"], axis=1)
    all_ohlcv = all_ohlcv.set_index(['code', 'date']).sort_index()
    idx_slice = pd.IndexSlice
    stk_ohlcv_list = []
    for stk in all_ohlcv.index.get_level_values('code').unique():
        if stk in stk_list:
            stk_ohlcv = all_ohlcv.loc[idx_slice[stk, :], idx_slice[:]]
            stk_ohlcv_list.append(stk_ohlcv)
    return stk_ohlcv_list


def valid_wave_by_multi_processes(params, ohlcv_list, operation, mode, processes=0):
    stk_result_list = []

    if processes <= 0:
        processes = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=processes)

    def callback(result):
        stk_result_list.append(result)

    def print_error(err):
        print(err)

    for ohlcv in ohlcv_list:
        if 'std_window' in params:
            std_window = params['std_window']
        else:
            std_window = 0
        pool.apply_async(tag_wave_direction, args=(
            ohlcv, params['max_return_threshold'], params['return_per_count_threshold'],
            params['withdraw_threshold'], params['minimum_period'], operation, mode, std_window,),
                         callback=callback, error_callback=print_error)

    pool.close()
    pool.join()
    return stk_result_list


def objective(params, ohlcv_list, operation, mode, log_dir):
    print(params)
    identity = str(uuid.uuid1())
    result_list = valid_wave_by_multi_processes(params, ohlcv_list, operation, mode)
    returns_list = []
    for result in result_list:
        stk_returns = result['pct_chg'] * result['direction']
        stk_returns = stk_returns.fillna(0)
        returns_list.append(stk_returns)
    returns = pd.concat(returns_list, axis=0)

    annual_return = empyrical.annual_return(returns)
    sharpe_ratio = empyrical.sharpe_ratio(returns)
    if np.isnan(sharpe_ratio):
        sharpe_ratio = 0

    data = {
        'id': identity,
        'params': params,
        'returns': returns,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'result_list': result_list
    }

    with open(os.path.join(log_dir, identity + '.pkl'), 'wb') as f:
        pickle.dump(data, f)

    print('id: %s, annual_return: %s, sharpe_ratio: %s' % (identity, annual_return, sharpe_ratio))
    return {'loss': -sharpe_ratio, 'status': STATUS_OK}


if __name__ == '__main__':
    space = relative_spaces
    sub_dir = 'relative'

    # function = tag_wave_direction_by_absolute()
    # space = absolute_spaces
    # sub_dir = 'absolute'

    ohlcv_list = get_data(zz500[:50])
    split_dates = ["2016-01-01", "2017-01-01"]
    train_set, validate_set, test_set = split_data_set_by_date(ohlcv_list, split_dates, minimum_size=1)
    log_dir = os.path.join('./valid_wave_hyperopt', sub_dir)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    hyperopt_objective = partial(objective, ohlcv_list=test_set, operation='search', mode='relative', log_dir=log_dir)
    trials = Trials()
    # best = fmin(hyperopt_objective, space, algo=tpe.suggest, max_evals=60, trials=trials)
    # params = space_eval(space, best)
    # print(params)

    best_params = {
        'minimum_period': 12.135881002390583,
        'std_window': 10.0,
        'withdraw_threshold': 2.0,
        'max_return_threshold': 1.0,
        'return_per_count_threshold': 1.0}

    hyperopt_objective(best_params)
