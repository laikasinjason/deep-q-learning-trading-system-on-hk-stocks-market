import numpy as np
import pandas as pd

bins = [-np.inf, -0.1, -0.05, -0.03, 0, 0.03, 0.05, 0.1, np.inf]
names = ['<-0.1', '-0.1--0.05', '-0.05--0.03', ' -0.03-0', '0-0.03', '0.03-0.05', '0.05-0.1', '>0.1']
no_data_to_remove = 15


def load_data(csv_file):
    if csv_file is None:
        return None
    data = pd.read_csv("../../HKDailyStocksQuotes/" + csv_file, index_col='Date')
    data.index = pd.to_datetime(data.index, format="%Y%m%d")
    return data


def turning_points(array):
    """ turning_points(array) -> min_indices, max_indices
    Finds the turning points within an 1D array and returns the indices of the minimum and
    maximum turning points in two separate lists.
    """
    idx_max, idx_min = [], []
    if len(array) < 3:
        return idx_min, idx_max

    NEUTRAL, RISING, FALLING = range(3)

    def get_state(a, b):
        if a < b:
            return RISING
        if a > b:
            return FALLING
        return NEUTRAL

    ps = get_state(array[0], array[1])
    begin = 1
    for i in range(2, len(array)):
        s = get_state(array[i - 1], array[i])
        if s != NEUTRAL:
            if ps != NEUTRAL and ps != s:
                if s == FALLING:
                    idx_max.append((begin + i - 1) // 2)
                else:
                    idx_min.append((begin + i - 1) // 2)
            begin = i
            ps = s
    return idx_min, idx_max


def sma(data, period=5):
    return data.rolling(period).mean()


def get_next_day(date, data):
    # next row in data
    next_index = data.index.get_loc(date) + 1
    if next_index >= len(data.index):
        return None
    else:
        return data.index[next_index]


def get_profit(data, buy_price):
    # used for sell signal agent 
    # 100×(closing price of the current day - buy price)/buy price
    return (data['Close'] - buy_price) / buy_price


def create_turning_point_3d_matrix(data):
    if data is None:
        return None, None
    # create turning points series
    idx_min, idx_max = turning_points(data['Adj Close'])

    # remove the first 15 rows explicitly
    max_matrix2 = create_turning_point_matrix_for_day_diff(data, 2, idx_max)[no_data_to_remove:]
    max_matrix3 = create_turning_point_matrix_for_day_diff(data, 3, idx_max)[no_data_to_remove:]
    max_matrix5 = create_turning_point_matrix_for_day_diff(data, 5, idx_max)[no_data_to_remove:]
    max_matrix10 = create_turning_point_matrix_for_day_diff(data, 10, idx_max)[no_data_to_remove:]
    max_matrix15 = create_turning_point_matrix_for_day_diff(data, 15, idx_max)[no_data_to_remove:]

    result_max = pd.concat([max_matrix2, max_matrix3, max_matrix5, max_matrix10, max_matrix15], keys=[2, 3, 5, 10, 15])
    result_max.index = result_max.index.swaplevel(1, 0)
    result_max = result_max.sort_index()
    result_max.index.names = ['Date', 'DayDiff']
    result_max.fillna(0)

    # remove the first 15 rows explicitly
    min_matrix2 = create_turning_point_matrix_for_day_diff(data, 2, idx_min)[no_data_to_remove:]
    min_matrix3 = create_turning_point_matrix_for_day_diff(data, 3, idx_min)[no_data_to_remove:]
    min_matrix5 = create_turning_point_matrix_for_day_diff(data, 5, idx_min)[no_data_to_remove:]
    min_matrix10 = create_turning_point_matrix_for_day_diff(data, 10, idx_min)[no_data_to_remove:]
    min_matrix15 = create_turning_point_matrix_for_day_diff(data, 15, idx_min)[no_data_to_remove:]

    result_min = pd.concat([min_matrix2, min_matrix3, min_matrix5, min_matrix10, min_matrix15], keys=[2, 3, 5, 10, 15])
    # rotate the matrix axes
    result_min.index = result_min.index.swaplevel(1, 0)
    result_min = result_min.sort_index()
    result_min.index.names = ['Date', 'DayDiff']
    result_min.fillna(0)

    return result_max, result_min


def create_turning_point_matrix_for_day_diff(data, day_diff, id_tp_array):
    # shift the turning point array to certain day difference, with the close price of turning point
    id_tp_array = pd.DataFrame({'max': [int(i in id_tp_array) for i in range(len(data))]})
    id_tp_array = id_tp_array.set_index(data.index)
    idx_tp_shift = id_tp_array.shift(day_diff)
    px_shift = data['Adj Close'].shift(day_diff)
    # cal px diff of Tday's px and TP's px
    px_diff = ((data['Adj Close'] - px_shift) / px_shift).mul(idx_tp_shift['max'], fill_value=0)

    # binning and transform to one hot categorization
    # fibonacci_bins = [-np.inf, -0.764, -0.618, -0.5, -0.382, 0, 0.382, 0.5, 0.618, 0.764, np.inf]
    px_diff_bin = pd.cut(px_diff, bins, labels=names)
    px_diff_bin = pd.get_dummies(px_diff_bin)
    px_diff_bin[px_diff == 0] = 0  # px-diff==0 is matched into 1 bin, but it should be zero in all cols

    return px_diff_bin


def create_technical_indicator_3d_matrix(data):
    if data is None:
        return None
    high_low_diff = (data['High'] - data['Low']) / data['Low']
    ma5_diff = (sma(data['Close'], 5) - data['Close']) / data['Close']
    ma10_diff = (sma(data['Close'], 10) - data['Close']) / data['Close']
    ma20_diff = (sma(data['Close'], 20) - data['Close']) / data['Close']

    high_low_diff_bin = pd.cut(high_low_diff, bins, labels=names)
    ma5_diff_bin = pd.cut(ma5_diff, bins, labels=names)
    ma10_diff_bin = pd.cut(ma10_diff, bins, labels=names)
    ma20_diff_bin = pd.cut(ma20_diff, bins, labels=names)

    result_hl_diff = pd.get_dummies(high_low_diff_bin)
    result_ma5_diff = pd.get_dummies(ma5_diff_bin)
    result_ma10_diff = pd.get_dummies(ma10_diff_bin)
    result_ma20_diff = pd.get_dummies(ma20_diff_bin)
    result_hl_diff[high_low_diff == 0] = 0  # remove 0 value
    result_ma5_diff[ma5_diff == 0] = 0  # remove 0 value
    result_ma10_diff[ma10_diff == 0] = 0  # remove 0 value
    result_ma20_diff[ma20_diff == 0] = 0  # remove 0 value
    result_hl_diff = result_hl_diff[no_data_to_remove:]
    result_ma5_diff = result_hl_diff[no_data_to_remove:]
    result_ma10_diff = result_hl_diff[no_data_to_remove:]
    result_ma20_diff = result_hl_diff[no_data_to_remove:]

    result_matrix = pd.concat([result_hl_diff, result_ma5_diff, result_ma10_diff, result_ma20_diff],
                              keys=['hl_diff', 'ma5_diff', 'ma10_diff', 'ma20_diff'])
    # rotate result matrix axes
    result_matrix.index = result_matrix.index.swaplevel(1, 0)
    result_matrix = result_matrix.sort_index()
    result_matrix.index.names = ['Date', 'Indicator']

    return result_matrix


def enrich_market_data(data):
    if data is None:
        return

    data['ma5'] = sma(data['Close'], 5)
    data['rate_of_close'] = data['Close'].pct_change()
    return data[no_data_to_remove:]
    
def clean_data(data):
    if data is None:
        return
    data = data.dropna()
    return data
