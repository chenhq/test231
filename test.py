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
ohlcv_all = all_ohlcv[all_ohlcv["code"] == "000725.XSHE"].drop("code", axis=1)

ohlcv = ohlcv_all.head(len(ohlcv_all) - 256).copy()
ohlcv_test = ohlcv_all.tail(256)

features = construct_features(ohlcv, construct_features1)

features_categorical, reverse_func = to_categorical(features.copy(), 'label', categorical_func_factory)

space = default_space

performance_measure = performance_factory(reverse_func, performace_type='annual_return')

objective_func = construct_objective1(features_categorical, "logs", performance_measure, loops=3)
trials = Trials()
best = fmin(objective_func, space, algo=tpe.suggest, max_evals=40, trials=trials)
print(best)
# best = {'batch_size': 0, 'dropout': 3, 'epochs': 1, 'initializer': 0,
# 'is_BN_1': 0, 'is_BN_2': 0, 'is_BN_3': 1, 'lr': 0.00026515359970862435,
# 'recurrent_dropout': 5, 'time_steps': 0, 'units1': 0, 'units2': 3, 'units3': 3}
# {'batch_size': 2, 'dropout': 2, 'epochs': 1, 'initializer': 0, 'is_BN_1': 0, 'is_BN_2': 1, 'is_BN_3': 0,
# 'lr': 0.00014243352196987224, 'recurrent_dropout': 2, 'time_steps': 1, 'units1': 3, 'units2': 2, 'units3': 0}
# {'batch_size': 8, 'dropout': 0.2, 'epochs': 200,
#  'initializer': <keras.initializers.VarianceScaling object at 0x000001C5C83E4B00>,
# 'is_BN_1': False, 'is_BN_2': True, 'is_BN_3': False, 'lr': 0.00014243352196987224,
# 'recurrent_dropout': 0.2, 'time_steps': 16, 'units1': 64, 'units2': 32, 'units3': 8}

params = space_eval(space, best)
for i in range(10):
    objective_func(params)
