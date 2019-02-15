import math

from agent import Agent
from model import Model


class SellOrderAgent(Agent):
    def __init__(self, environment, buy_signal_agent=None):
        super().__init__(environment)

        self.bp = None
        # high turning point 5*8, low turning point 5*8, technical indicator 4*8, profit 8
        self.model = Model(7, 120)
        self.__buy_signal_agent = buy_signal_agent
        self.state = None # save the state to be trained
        self.action = None # save the action needed to pass to fit method

    def get_buy_signal_agent(self):
        return self.__buy_signal_agent

    def set_buy_signal_agent(self, buy_signal_agent):
        self.__buy_signal_agent = buy_signal_agent

    def process_next_state(self):
        self.state, action = self.produce_state_and_get_action()
        self.process_action(action)

    def process_action(self, action):
        ma5 = 4
        high = 1
        d = ma5 + action / 100 * ma5 - high
        print("processing sell order")

        if d <= 0:
            reward = math.exp(100 * d / high)
            sp = ma5 + action / 100 * ma5
            self.model.fit(self.state, reward, action)

            self.invoke_buy_signal_agent(sp)

        else:
            reward = 0
            self.model.fit(self.state, reward, action)

            close = 3
            sp = close
            self.invoke_buy_signal_agent(sp)

    def invoke_buy_signal_agent(self, sp):
        self.__buy_signal_agent.update_reward(False, self.bp, sp)

    def start_new_training(self, bp):
        print("Sell order - start new training")
        self.bp = bp
        self.process_next_state()
