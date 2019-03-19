import math

from agent import Agent
from model import OrderModel


class BuyOrderAgent(Agent):
    def __init__(self, environment, buy_signal_agent=None, sell_signal_agent=None):
        super().__init__(environment)

        # technical indicator 4*8
        self.model = OrderModel(7, 32)
        self.__buy_signal_agent = buy_signal_agent
        self.__sell_signal_agent = sell_signal_agent
        self.state = None  # save the state to be trained

    def get_buy_signal_agent(self):
        return self.__buy_signal_agent

    def set_buy_signal_agent(self, buy_signal_agent):
        self.__buy_signal_agent = buy_signal_agent

    def get_sell_signal_agent(self):
        return self.__sell_signal_agent

    def set_sell_signal_agent(self, sell_signal_agent):
        self.__sell_signal_agent = sell_signal_agent

    def process_action(self, action, last_state_date):
        # buy order agent consider state on T-1, and place order on T day

        next_date = self.environment.get_next_day_of_state(last_state_date)
        if next_date is None:
            # not able to get next date's market data
            return False

        market_data = self.environment.get_market_data_by_date(next_date)

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
        print("processing buy order, buy price: " + str(bp))

        if d >= 0:
            reward = math.exp(-100 * d / low)

            # if not self.environment.get_evaluation_mode():
            # self.model.fit(self.state.value, reward, action)
            # else:
            # record = {'bp' : bp, 'date' : next_date}
            # self.environment.record(record)

            # last state date for sell signal becomes T day, start training on T+1

            self.invoke_sell_signal_agent(bp, next_date)
        else:
            reward = 0
            # if not self.environment.get_evaluation_mode():
            # self.model.fit(self.state.value, reward, action)
            self.invoke_buy_signal_agent()
        return True

    def invoke_sell_signal_agent(self, bp, last_state_date):
        self.__sell_signal_agent.start_new_training(bp, last_state_date)

    def invoke_buy_signal_agent(self):
        self.__buy_signal_agent.update_reward(True, self.state.date)

    def start_new_training(self, date):
        print("Buy order - start new training " + str(date))
        self.state = self.environment.get_buy_order_states_by_date(date)
        action = self.get_action(self.state)

        result = self.process_action(action, date)
        if not result:
            self.environment.terminate_epoch()
