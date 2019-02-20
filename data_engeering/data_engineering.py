def turning_points(array):
    ''' turning_points(array) -> min_indices, max_indices
    Finds the turning points within an 1D array and returns the indices of the minimum and 
    maximum turning points in two separate lists.
    '''
    idx_max, idx_min = [], []
    if (len(array) < 3): 
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
    
def get_rate_of_change_of_close(data):
    # used for the reward training on sell signal agent
    data['rc'] = data['close'].pct_change()

def get_profit(data, buy_price):
    # used for sell signal agent 
    # 100×(closing price of the current day - buy price)/buy price
    return (data['close'] - buy_price)/buy_price
    
def create_turning_point_matrix(data):
    # create turning points series
    idx_min, idx_max = turning_points(array)
    
    max_matrix = create_turning_point_series(data, 2, idx_max)
    max_matrix = pd.concat([max_matrix, create_turning_point_series(data, 3, idx_max)], axis=1)
    max_matrix = pd.concat([max_matrix, create_turning_point_series(data, 5, idx_max)], axis=1)
    max_matrix = pd.concat([max_matrix, create_turning_point_series(data, 10, idx_max)], axis=1)
    max_matrix = pd.concat([max_matrix, create_turning_point_series(data, 15, idx_max}], axis=1)
    
    min_matrix = create_turning_point_series(data, 2, idx_min)
    min_matrix = pd.concat([min_matrix, create_turning_point_series(data, 3, idx_min)], axis=1)
    min_matrix = pd.concat([min_matrix, create_turning_point_series(data, 5, idx_min)], axis=1)
    min_matrix = pd.concat([min_matrix, create_turning_point_series(data, 10, idx_min)], axis=1)
    min_matrix = pd.concat([min_matrix, create_turning_point_series(data, 15, idx_min}], axis=1)


    
def create_turning_point_series(data, day_diff, id_tp_array):
     # shift the turning point array to certain day difference, with the close price of turning point
    idx_tp_shift = id_tp_array.shift(day_diff)
    px_shift = data['close'].shift(day_diff)
    # cal px diff of Tday's px and TP's px
    px_diff = ((data['close'] - px_shift)/px_shift).where(idx_tp_shift==1)
    # binning and transform to one hot categorization
    bins = [-np.inf, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, np.inf]
    # fibonacci_bins = [-np.inf, -0.764, -0.618, -0.5, -0.382, 0, 0.382, 0.5, 0.618, 0.764, np.inf]
    names = ['<-0.2', '-0.2--0.1', '-0.1--0.05',' -0.05-0', '0-0.05', '0.05-0.1', '0.1-0.2', '>0.2']
    px_diff_bin = pd.cut(px_diff, bins, labels=names)
    
    return pd.get_dummies(px_diff_bin)
    