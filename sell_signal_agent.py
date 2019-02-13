from agent import Agent
from model import Model


class SellSignalAgent(Agent):
    def __init__(self, sell_order_agent=None):
        self.BP = None
        self.__model = Model(2, 50)
        self.__sell_order_agent = sell_order_agent

    def get_sell_order_agent(self):
        return self.__sell_order_agent

    def set_sell_order_agent(self, sell_order_agent):
        self.__sell_order_agent = sell_order_agent

    def process_action(self, sell_action):
        if not sell_action:
            rc = 1
            r = rc
            self.process_next_state()
        else:
            self.invoke_sell_order_agent()

    def process_next_state(self):
        sell_action = self.produce_state_and_get_action()
        self.process_action(sell_action)

    def invoke_sell_order_agent(self):
        self.__sell_order_agent.start_new_training(self.BP)

    def start_new_training(self, bp):
        print("Sell signal - start new training")
        self.BP = bp
        self.process_next_state()
