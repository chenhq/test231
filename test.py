import numpy as np
import pandas as pd
from data_prepare import *
import matplotlib.pyplot as plt
from params_select import *
import hyperopt.pyll.stochastic
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial, rand, space_eval

# market = pd.read_csv("~/cs_market.csv", parse_dates=["date"], dtype={"code": str})
market = pd.read_csv("E:\market_data/cs_market.csv", parse_dates=["date"], dtype={"code": str})
all_ohlcv = market.drop(["Unnamed: 0", "total_turnover", "limit_up", "limit_down"], axis=1)
all_ohlcv = all_ohlcv.set_index('date')
ohlcv = all_ohlcv[all_ohlcv["code"] == "000725.XSHE"].drop("code", axis=1)

features = construct_features(ohlcv, construct_features1)

features_categorical, reverse_func = to_categorical(features.copy(), 'label', categorical_func_factory)

space = default_space

performance_return = performance_factory(reverse_func)

objective_func = construct_objective1(features_categorical, "logs", performance_return, loops=2)
trials = Trials()
best = fmin(objective_func, space, algo=tpe.suggest, max_evals=40, trials=trials)
print(best)

# best = {'batch_size': 0, 'dropout': 3, 'epochs': 1, 'initializer': 0, 'is_BN_1': 0, 'is_BN_2': 0, 'is_BN_3': 1, 'lr': 0.00026515359970862435, 'recurrent_dropout': 5, 'time_steps': 0, 'units1': 0, 'units2': 3, 'units3': 3}
params = space_eval(space, best)
for i in range(10):
    objective_func(params)



