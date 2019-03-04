import math

from agent import Agent
from model import Model


class BuyOrderAgent(Agent):
    def __init__(self, environment, buy_signal_agent=None, sell_signal_agent=None):
        super().__init__(environment)

        # technical indicator 4*8
        self.model = Model(7, 32)
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
        market_data = self.environment.get_market_data_by_date_of_state(last_state_date + 1)

        if market_data is None:
            # terminate
            return True

        ma5 = market_data['ma5']
        low = market_data['low']

        if ma5 is None or low is None:
            # terminate
            return True

        d = ma5 + action / 100 * ma5 - low
        print(action)
        print("processing buy order")

        if d >= 0:
            reward = math.exp(-100 * d / low)
            bp = ma5 + action / 100 * ma5
            # self.model.fit(self.state, reward, action)
            # last state date for sell signal becomes T day, start training on T+1
            self.invoke_sell_signal_agent(bp, last_state_date + 1)
        else:
            reward = 0
            # self.model.fit(self.state, reward, action)
            self.invoke_buy_signal_agent()
        return False

    def invoke_sell_signal_agent(self, bp, last_state_date):
        self.__sell_signal_agent.start_new_training(bp, last_state_date)

    def invoke_buy_signal_agent(self):
        self.__buy_signal_agent.update_reward(True)

    def restart_training(self):
        # state is not available, restart from the top
        self.__buy_signal_agent.start_new_training()

    def start_new_training(self, state):
        print("Buy order - start new training")
        action = self.get_action(state)
        this_state_date = state['date']  # get state's date
        terminated = self.process_action(action, this_state_date)

        if terminated:
            self.restart_training()
