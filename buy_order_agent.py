import math

from agent import Agent
from model import OrderModel


class BuyOrderAgent(Agent):
    state = None  # save the state to be trained

    def __init__(self, environment):
        super().__init__(environment)
        # technical indicator 4*8
        self.model = OrderModel(7, [32], 50, str(self.__class__.__name__))

    def process_action(self, action, date):
        # buy order agent consider state on T-1, and place order on T day
        market_data = self.environment.get_market_data_by_date(date)

        if market_data is None:
            # state is not available, restart from the top
            return False

        ma5 = market_data['ma5']
        low = market_data['Low']

        if ma5 is None or low is None:
            # terminate
            return False

        bp = ma5 + action / 100 * ma5
        d = bp - low
        # print("processing buy order, buy price: " + str(bp))

        if not self.environment.get_evaluation_mode():
            self.fit_all_actions(ma5, low)

        if d >= 0:
            if self.environment.get_evaluation_mode():
                record = {'bp': bp, 'date': date}
                self.environment.record(**record)

            # last state date for sell signal becomes T day, start training on T+1
            self.environment.set_buy_price(bp)
            self.environment.invoke_sell_signal_agent()
        else:
            self.environment.invoke_buy_signal_agent(True, self.state.date)
        return True

    def process_next_state(self, date):
        # the date get here is already the next day, but we need the same day of BSA as the state
        prev_date = self.environment.get_prev_day(date)
        # print("Buy order - processing date: " + str(date))
        self.state = self.environment.get_buy_order_states_by_date(prev_date)
        action = self.get_action(self.state)

        result = self.process_action(action, date)
        if not result:
            self.environment.process_epoch_end(None, True)

    def fit_all_actions(self, ma5, low):
        # rewards are well-defined, we generate all rewards for every possible actions and fit the model
        for action in OrderModel.action_map.keys():
            bp = ma5 + action / 100 * ma5
            d = bp - low
            if d >= 0:
                reward = math.exp(-100 * d / low)
            else:
                reward = 0
            self.model.fit(self.state.value, reward, action)
