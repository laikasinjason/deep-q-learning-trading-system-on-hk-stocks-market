import math

from agent import Agent
from model import Model


class BuyOrderAgent(Agent):
    def __init__(self, environment, buy_signal_agent=None, sell_signal_agent=None):
        super().__init__(environment)

        self.model = Model(7, 50)
        self.__buy_signal_agent = buy_signal_agent
        self.__sell_signal_agent = sell_signal_agent
        self.state = None

    def get_buy_signal_agent(self):
        return self.__buy_signal_agent

    def set_buy_signal_agent(self, buy_signal_agent):
        self.__buy_signal_agent = buy_signal_agent

    def get_sell_signal_agent(self):
        return self.__sell_signal_agent

    def set_sell_signal_agent(self, sell_signal_agent):
        self.__sell_signal_agent = sell_signal_agent

    def process_action(self, action):
        market_data = self.environment.get_market_data_by_date_of_state(self.state)

        ma5 = market_data['ma5']
        low = market_data['low']
        d = ma5 + action / 100 * ma5 - low
        print("processing buy order")

        if d >= 0:
            reward = math.exp(-100 * d / low)
            bp = ma5 + action / 100 * ma5
            self.model.fit(self.state, reward)
            self.invoke_sell_signal_agent(bp)
        else:
            reward = 0
            self.model.fit(self.state, reward)
            self.invoke_buy_signal_agent()

    def process_next_state(self):
        self.state, new_action = self.produce_state_and_get_action()
        self.process_action(new_action)

    def invoke_sell_signal_agent(self, bp):
        self.__sell_signal_agent.start_new_training(bp)

    def invoke_buy_signal_agent(self):
        self.__buy_signal_agent.update_reward(False)

    def start_new_training(self):
        print("Buy order - start new training")
        self.process_next_state()
