import numpy as np
import matplotlib.pylab as plt


# operation='search'/'label'
# mode='relative'/'absolute'
def tag_wave_direction(ohlcv, max_return_threshold, return_per_count_threshold, withdraw_threshold, minimum_period,
                       operation='search', mode='relative', std_window=30):
    ohlcv['pct_chg'] = (ohlcv['close'] / ohlcv['close'].shift(1) - 1).fillna(0)
    ohlcv['std'] = ohlcv['pct_chg'].rolling(int(std_window)).std().bfill(0)
    ohlcv['direction'] = np.nan
    i = 0
    while i < len(ohlcv):
        # 波段的周期数量
        count = 1
        price = ohlcv.iloc[i]['open']
        if ohlcv.iloc[i]['close'] > ohlcv.iloc[i]['open']:
            min_price = ohlcv.iloc[i]['open']
            max_price = ohlcv.iloc[i]['close']
            direction = 1
        elif ohlcv.iloc[i]['close'] < ohlcv.iloc[i]['open']:
            min_price = ohlcv.iloc[i]['close']
            max_price = ohlcv.iloc[i]['open']
            direction = -1
        else:
            i += 1
            continue

        argmin = i
        argmax = i
        # print(ohlcv.index[i], direction, price)
        j = i + 1
        if j >= len(ohlcv):
            i += 1
            continue
        while j < len(ohlcv):
            count += 1
            new_price = ohlcv.iloc[j]['close']

            if new_price > max_price:
                max_price = new_price
                argmax = j

            if new_price < min_price:
                min_price = new_price
                argmin = j

            returns = (new_price / price - 1) * direction
            return_per_count = returns / count

            if direction == 1:
                max_return = (max_price / price - 1)
                withdraw = (max_price - new_price) / price
                max_return_pos = argmax
            elif direction == -1:
                max_return = (1 - min_price / price)
                withdraw = (new_price - min_price) / price
                max_return_pos = argmin

            if mode == 'relative':
                return_per_count_threshold_ = return_per_count_threshold * ohlcv.iloc[j]['std']
                max_return_threshold_ = max_return_threshold * ohlcv.iloc[j]['std']
                withdraw_threshold_ = withdraw_threshold * ohlcv.iloc[j]['std']
            else:
                return_per_count_threshold_ = return_per_count_threshold
                max_return_threshold_ = max_return_threshold
                withdraw_threshold_ = withdraw_threshold

            # 波段趋势太小，未继续满足return_per_count_threshold_而结束
            if return_per_count < return_per_count_threshold_:
                wave_end_pos = i
                # 波段满足max_return_threshold or minimum period
                if max_return > max_return_threshold_ or max_return_pos - i >= minimum_period:
                    # 确定最后的标签位置
                    if operation == 'search':
                        wave_end_pos = j
                    elif operation == 'label':
                        wave_end_pos = max_return_pos

                    # 符合条件的波段
                    # 标记[i, wave_end_pos]的direction标签
                    for k in range(i, wave_end_pos + 1):
                        ohlcv.loc[ohlcv.index[k], 'direction'] = direction
                i = wave_end_pos + 1
                break

            # 波段因为最大回撤结束
            if withdraw > withdraw_threshold_:
                wave_end_pos = i
                # 满足max_return_threshold or minimum period
                if max_return > max_return_threshold_ or max_return_pos - i >= minimum_period:
                    # 确定最后的标签位置
                    if operation == 'search':
                        wave_end_pos = j
                    elif operation == 'label':
                        wave_end_pos = max_return_pos

                    # 符合条件的波段
                    # 标记[i, wave_end_pos]的direction标签
                    for k in range(i, wave_end_pos + 1):
                        ohlcv.loc[ohlcv.index[k], 'direction'] = direction
                i = max(wave_end_pos+1, i+1)
                break
            j += 1
            if j >= len(ohlcv):
                i += 1
    # # show
    # result = ohlcv.reset_index().reset_index()
    # fig, ax = plt.subplots(1, figsize=(21, 7))
    # result.plot(x='index', y='close', figsize=(21, 7), ax=ax)
    # result[result['direction'] > 0].plot.scatter(x='index', y='close', s=10, c='r', figsize=(21, 7), ax=ax)
    # result[result['direction'] < 0].plot.scatter(x='index', y='close', s=10, c='g', figsize=(21, 7), ax=ax)
    # result[np.isnan(result['direction'])].plot.scatter(x='index', y='close', s=10, c='b', figsize=(21, 7), ax=ax)
    # plt.show()
    return ohlcv
