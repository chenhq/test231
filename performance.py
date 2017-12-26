import empyrical
import pandas as pd


def performance_factory(reverse_func, performance_types=['returns']):
    def performance_measures(pct_chg, y):
        result = {}
        y_init = list(map(reverse_func, y))
        predict = pd.Series(index=pct_chg.index, data=y_init)
        predict.name = 'label'
        df = pd.concat([pct_chg, predict.shift(1)], axis=1)
        df['return'] = 0
        epsilon = 0.0001
        long_cond = (abs(df['label'] - 2)) < epsilon
        short_cond = (abs(df['label'])) < epsilon
        df.loc[long_cond, 'return'] = pct_chg.loc[long_cond]/100.0
        df.loc[short_cond, 'return'] = -pct_chg[short_cond]/100.0
        returns = df['return']

        if 'Y0' in performance_types:
            Y0 = pd.Series(index=pct_chg.index, data=list(y))
            result['Y0'] = Y0
        if 'Y' in performance_types:
            result['Y'] = predict
        if 'returns' in performance_types:
            result['returns'] = returns
        if 'cum_returns' in performance_types:
            result['cum_returns'] = empyrical.cum_returns(returns)
        if 'annual_return' in performance_types:
            result['annual_return'] = empyrical.annual_return(returns)
        if 'sharpe_ratio' in performance_types:
            result['sharpe_ratio'] = empyrical.sharpe_ratio(returns)
        return result
    return performance_measures

