import numpy as np
import pandas as pd

bins = [-np.inf, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, np.inf]
names = ['<-0.2', '-0.2--0.1', '-0.1--0.05', ' -0.05-0', '0-0.05', '0.05-0.1', '0.1-0.2', '>0.2']


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
        if a < b: return RISING
        if a > b: return FALLING
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


def get_yesterday_sma():
    # the buy/sell price of order agent is determined by last day's sma
    pass


def get_today_close():
    # used to close the position at day end if sell price is not met
    pass


def get_today_low():
    # used to determined if buy order is successfully executed
    pass


def get_today_high():
    # used to determined if sell order is successfully executed
    pass


def get_profit(data, buy_price):
    # used for sell signal agent 
    # 100×(closing price of the current day - buy price)/buy price
    return (data['close'] - buy_price) / buy_price


def create_turning_point_4d_matrix(data):
    if data is None:
        return None, None
    # create turning points series
    idx_min, idx_max = turning_points(data['close'])

    max_matrix = create_turning_point_matrix_for_day_diff(data, 2, idx_max)
    max_matrix = pd.concat([max_matrix, create_turning_point_matrix_for_day_diff(data, 3, idx_max)], axis=1)
    max_matrix = pd.concat([max_matrix, create_turning_point_matrix_for_day_diff(data, 5, idx_max)], axis=1)
    max_matrix = pd.concat([max_matrix, create_turning_point_matrix_for_day_diff(data, 10, idx_max)], axis=1)
    max_matrix = pd.concat([max_matrix, create_turning_point_matrix_for_day_diff(data, 15, idx_max)], axis=1)

    min_matrix = create_turning_point_matrix_for_day_diff(data, 2, idx_min)
    min_matrix = pd.concat([min_matrix, create_turning_point_matrix_for_day_diff(data, 3, idx_min)], axis=1)
    min_matrix = pd.concat([min_matrix, create_turning_point_matrix_for_day_diff(data, 5, idx_min)], axis=1)
    min_matrix = pd.concat([min_matrix, create_turning_point_matrix_for_day_diff(data, 10, idx_min)], axis=1)
    min_matrix = pd.concat([min_matrix, create_turning_point_matrix_for_day_diff(data, 15, idx_min)], axis=1)

    # rotate the matrix axes
    return max_matrix, min_matrix


def create_turning_point_matrix_for_day_diff(data, day_diff, id_tp_array):
    # shift the turning point array to certain day difference, with the close price of turning point
    id_tp_array = pd.Series([int(i in id_tp_array) for i in range(len(data))])
    idx_tp_shift = id_tp_array.shift(day_diff)
    px_shift = data['Adj Close'].shift(day_diff)
    # cal px diff of Tday's px and TP's px
    px_diff = ((data['Adj Close'] - px_shift) / px_shift).where(idx_tp_shift == 1)
    # binning and transform to one hot categorization
    # fibonacci_bins = [-np.inf, -0.764, -0.618, -0.5, -0.382, 0, 0.382, 0.5, 0.618, 0.764, np.inf]
    px_diff_bin = pd.cut(px_diff, bins, labels=names)

    return pd.get_dummies(px_diff_bin)


def create_technical_indicator_4d_matrix(data):
    if data is None:
        return None
    high_low_diff = (data['high'] - data['low']) / data['low']
    ma5_diff = (sma(data['close'], 5) - data['close']) / data['close']
    ma10_diff = (sma(data['close'], 10) - data['close']) / data['close']
    ma20_diff = (sma(data['close'], 20) - data['close']) / data['close']
    
    high_low_diff_bin = pd.cut(high_low_diff, bins, labels=names)
    ma5_diff_bin = pd.cut(ma5_diff, bins, labels=names)
    ma10_diff_bin = pd.cut(ma10_diff, bins, labels=names)
    ma20_diff_bin = pd.cut(ma20_diff, bins, labels=names)

    result_matrix = pd.get_dummies(high_low_diff_bin)
    result_matrix = pd.concat([result_matrix, pd.get_dummies(ma5_diff_bin)], axis=1)
    result_matrix = pd.concat([result_matrix, pd.get_dummies(ma10_diff_bin)], axis=1)
    result_matrix = pd.concat([result_matrix, pd.get_dummies(ma20_diff_bin)], axis=1)

    # rotate result matrix axes
    return result_matrix


def enrich_market_data(data):
    if data is None:
        return

    data['ma5'] = sma(data['close'], 5)
    data['rate_of_close'] = data['close'].pct_change()
