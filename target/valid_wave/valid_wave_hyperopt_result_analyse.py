import matplotlib.pylab as plt
import _pickle as pickle
import os

import pandas as pd

columns = ['id', 'std_window', 'max_return_threshold', 'return_per_count_threshold', 'withdraw_threshold',
           'minimum_period', 'annual_return', 'sharpe_ratio']
results = pd.DataFrame(columns=columns)
log_dir = "./valid_wave_hyperopt"
for root, dirs, files in os.walk(log_dir, topdown=False):
    for name in files:
        if name.endswith('.pkl'):
            with open(os.path.join(root, name), 'rb') as f:
                result = pickle.load(f)
            results.loc[len(results)] = [result['id'], result['params']['std_window'],
                                         result['params']['max_return_threshold'],
                                         result['params']['return_per_count_threshold'],
                                         result['params']['withdraw_threshold'],
                                         result['params']['minimum_period'],
                                         result['annual_return'],
                                         result['sharpe_ratio']]

results = results.sort_values(['sharpe_ratio']).set_index('id')

for column in results.columns:
    results[column].plot(figsize=(21, 7), label=column)
    plt.legend()
    plt.show()

quantile = 0.8
tops = results[results.sharpe_ratio > results.sharpe_ratio.quantile(quantile)]
tops_mean = tops.mean(axis=0)
print(tops_mean)

