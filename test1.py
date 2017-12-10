import numpy as np
import pandas as pd
from data_prepare import *
import matplotlib.pyplot as plt
from params_select import *
import hyperopt.pyll.stochastic
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial, rand, space_eval


market = pd.read_csv("~/cs_market.csv", parse_dates=["date"], dtype={"code": str})
# market = pd.read_csv("E:\market_data/cs_market.csv", parse_dates=["date"], dtype={"code": str})
all_ohlcv = market.drop(["Unnamed: 0", "total_turnover", "limit_up", "limit_down"], axis=1)

all_ohlcv = all_ohlcv.set_index(['code', 'date']).sort_index()

idx_slice = pd.IndexSlice

stk_ohlcv_dict = {}
for stk in all_ohlcv.index.get_level_values('code').unique():
    stk_ohlcv = all_ohlcv.loc[idx_slice[stk, :], idx_slice[:]]
    stk_ohlcv_dict[stk] = stk_ohlcv

stk_features_dict = construct_features_for_stocks(stk_ohlcv_dict, construct_features1)

new_stk_features_dict, reverse_func = data_set_to_categorical(stk_features_dict,
                                                              'label', categorical_func_factory)

split_dates = ["2016-01-01", "2017-03-01"]

train_set, validate_set, test_set = split_data_set_by_date(stk_features_dict, split_dates, minimum_size=128)

train = flatten_stock_features(train_set)
validate = flatten_stock_features(validate_set)
test = flatten_stock_features(test_set)

data_set = {'train': train, 'validate': validate, 'test': test}

space = default_space

performance_measure = performance_factory(reverse_func,
                                          performace_type=['cum_returns', 'annual_return', 'sharpe_ratio'])

objective_func = construct_objective2(data_set, "logs", performance_measure, 'sharpe_ratio')


trials = Trials()
best = fmin(objective_func, space, algo=tpe.suggest, max_evals=40, trials=trials)
print(best)

# params = space_eval(space, best)
# for i in range(10):
#     objective_func(params)
