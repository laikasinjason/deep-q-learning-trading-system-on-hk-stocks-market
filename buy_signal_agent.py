from agent import Agent
from environment import transaction_cost
from model import Model


class BuySignalAgent(Agent):
    def __init__(self, buy_order_agent=None):
        self.__model = Model(2, 50)
        self.__buy_order_agent = buy_order_agent

    def get_buy_order_agent(self):
        return self.__buy_order_agent

    def set_buy_order_agent(self, buy_order_agent):
        self.__buy_order_agent = buy_order_agent

    def process_next_state(self):
        buy_action = self.produce_state_and_get_action()
        self.process_action(buy_action)

    def process_action(self, buy_action):
        if not buy_action:
            self.process_next_state()
        else:
            self.invoke_buy_order_agent()

    def update_reward(self, from_but_order_agent, bp=None, sp=None):
        if from_but_order_agent:
            r = 0
        else:
            r = ((1 - transaction_cost) * sp - bp) / bp

    def invoke_buy_order_agent(self):
        self.__buy_order_agent.start_new_training()
