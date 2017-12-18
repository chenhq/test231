import numpy as np
import matplotlib.pylab as plt


def tag_wave_direction_by_absolute(ohlcv, max_return_threshold, return_per_count_threshold, withdraw_threshold):
    ohlcv['pct_chg'] = (ohlcv['close'] / ohlcv['close'].shift(1) - 1).fillna(0)
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

            # 波段趋势太小，未满足return_per_count_threshold
            if return_per_count < return_per_count_threshold:
                # 波段太小，未满足max_return_threshold
                if max_return < max_return_threshold:
                    i += 1
                    break
                else:
                    # 符合条件的波段
                    # 标记[i, max_return_pos]的direction标签
                    for k in range(i, max_return_pos + 1):
                        ohlcv.loc[ohlcv.index[k], 'direction'] = direction
                    i = max_return_pos + 1
                    break

            # 波段因为最大回撤结束
            if withdraw > withdraw_threshold:
                # 波段太小，未满足max_return_threshold
                if max_return < max_return_threshold:
                    i += 1
                    break
                else:
                    # 符合条件的波段
                    # 标记[i, max_return_pos]的direction标签
                    for k in range(i, max_return_pos + 1):
                        ohlcv.loc[ohlcv.index[k], 'direction'] = direction
                    i = max_return_pos + 1
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


def tag_wave_direction_by_relative(ohlcv, window, max_return_threshold, return_per_count_threshold, withdraw_threshold):
    ohlcv['pct_chg'] = (ohlcv['close'] / ohlcv['close'].shift(1) - 1).fillna(0)
    ohlcv['std'] = ohlcv['pct_chg'].rolling(int(window)).std().bfill(0)
    ohlcv['direction'] = np.nan
    print(ohlcv)
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

            # 波段趋势太小，未满足return_per_count_threshold
            if return_per_count < return_per_count_threshold * ohlcv.iloc[j]['std']:
                # 波段太小，未满足max_return_threshold
                if max_return < max_return_threshold * ohlcv.iloc[j]['std']:
                    i += 1
                    break
                else:
                    # 符合条件的波段
                    # 标记[i, max_return_pos]的direction标签
                    for k in range(i, max_return_pos + 1):
                        ohlcv.loc[ohlcv.index[k], 'direction'] = direction
                    i = max_return_pos + 1
                    break

            # 波段因为最大回撤结束
            if withdraw > withdraw_threshold * ohlcv.iloc[j]['std']:
                # 波段太小，未满足max_return_threshold
                if max_return < max_return_threshold * ohlcv.iloc[j]['std']:
                    i += 1
                    break
                else:
                    # 符合条件的波段
                    # 标记[i, max_return_pos]的direction标签
                    for k in range(i, max_return_pos + 1):
                        ohlcv.loc[ohlcv.index[k], 'direction'] = direction
                    i = max_return_pos + 1
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
