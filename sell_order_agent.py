import math

from agent import Agent
from model import Model


class SellOrderAgent(Agent):
    def __init__(self, model, buy_signal_agent=None):
        self.bp = None
        self.__model = Model(7, 50)
        self.__buy_signal_agent = buy_signal_agent

    def get_buy_signal_agent(self):
        return self.__buy_signal_agent

    def set_buy_signal_agent(self, buy_signal_agent):
        self.__buy_signal_agent = buy_signal_agent

    def process_next_state(self):
        action = self.produce_state_and_get_action()
        self.process_action(action)

    def process_action(self, action):
        ma5 = 4
        high = 1
        d = ma5 + action / 100 * ma5 - high
        print("processing sell order")

        if d <= 0:
            r = math.exp(100 * d / high)
            sp = ma5 + action / 100 * ma5

            self.invoke_buy_signal_agent(sp)

        else:
            r = 0

            close = 3
            sp = close
            self.invoke_buy_signal_agent(sp)

    def invoke_buy_signal_agent(self, sp):
        self.__buy_signal_agent.update_reward(False, self.bp, sp)

    def start_new_training(self, bp):
        print("Sell order - start new training")
        self.bp = bp
        self.process_next_state()
