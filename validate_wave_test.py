from params_select import *
from valid_wave import tag_wave_direction_by_absolute, tag_wave_direction_by_relative


market = pd.read_csv("E:\market_data/cs_market.csv", parse_dates=["date"], dtype={"code": str})
all_ohlcv = market.drop(["Unnamed: 0", "total_turnover", "limit_up", "limit_down"], axis=1)
all_ohlcv = all_ohlcv.set_index(['code', 'date']).sort_index()
idx_slice = pd.IndexSlice
stk_ohlcv_list = []
for stk in all_ohlcv.index.get_level_values('code').unique():
    stk_ohlcv = all_ohlcv.loc[idx_slice[stk, :], idx_slice[:]]
    stk_ohlcv_list.append(stk_ohlcv)

ohlcv = stk_ohlcv_list[0]

window = 30
max_return_threshold = 10
return_per_count_threshold = 0.02
withdraw_threshold = 4

x = tag_wave_direction_by_relative(ohlcv.copy(), window, max_return_threshold, return_per_count_threshold,
                                   withdraw_threshold)


