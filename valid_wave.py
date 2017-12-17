import numpy as np


def tag_wave_direction_by_absolute(ohlcv, max_return_threshold, return_per_count_threshold, withdraw_threshold):
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
        while j <= len(ohlcv):
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

            # print("i: %s, j: %s, direction: %d, price: %.2f, new_price: %.2f, max_price: %.2f, argmax: %d, argmax_date: %s, "
            #       + "min_price: %.2f, argmin: %d, argmin_date: %s, returns: %.2f, return_per_count: %.2f, max_return: %.2f, "
            #         + "max_return_pos: %s, max_return_pos_date: %s, withdraw: %.2f" %
            #       (ohlcv.index[i], ohlcv.index[j], direction, price, new_price, max_price, argmax, ohlcv.index[argmax],
            #       min_price, argmin, ohlcv.index[argmin], returns, return_per_count, max_return, max_return_pos,
            #       ohlcv.index[max_return_pos], withdraw))

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
                        ohlcv.iloc[k]['direction'] = direction
                        i = max_return_pos + 1
                        break
            else:
                j += 1

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
                        ohlcv.iloc[k]['direction'] = direction
                        i = max_return_pos + 1
                        break
            else:
                # 未达到最大回撤
                j += 1
    return ohlcv


def tag_wave_direction_by_relative(ohlcv, window, max_return_threshold, return_per_count_threshold, withdraw_threshold):
    ohlcv['pct_chg'] = ohlcv['close'] / ohlcv['close'].shift(-1) - 1
    print('window: %d' % window)
    ohlcv['std'] = ohlcv.rolling(int(window)).std()
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
        while j <= len(ohlcv):
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

            # print("i: %s, j: %s, direction: %d, price: %.2f, new_price: %.2f, max_price: %.2f, argmax: %d, argmax_date: %s, "
            #       + "min_price: %.2f, argmin: %d, argmin_date: %s, returns: %.2f, return_per_count: %.2f, max_return: %.2f, "
            #         + "max_return_pos: %s, max_return_pos_date: %s, withdraw: %.2f" %
            #       (ohlcv.index[i], ohlcv.index[j], direction, price, new_price, max_price, argmax, ohlcv.index[argmax],
            #       min_price, argmin, ohlcv.index[argmin], returns, return_per_count, max_return, max_return_pos,
            #       ohlcv.index[max_return_pos], withdraw))

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
                        ohlcv.iloc[k]['direction'] = direction
                        i = max_return_pos + 1
                        break
            else:
                j += 1

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
                        ohlcv.iloc[k]['direction'] = direction
                        i = max_return_pos + 1
                        break
            else:
                # 未达到最大回撤
                j += 1
    return ohlcv