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
    
def create_truning_point_matrix():
    # create turning points series
    idx_min, idx_max = turning_points(array)
    
    # shift the turning point array to certain day difference, with the close price of turning point
    idx_max_shift2 = idx_max.shift(2)
    # cal px diff of Tday's px and TP's px
    # binning and transform to one hot categorization