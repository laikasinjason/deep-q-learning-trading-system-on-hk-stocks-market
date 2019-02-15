from agent import Agent
from model import Model


class BuySignalAgent(Agent):
    def __init__(self, environment, no_train, buy_order_agent=None):
        super().__init__(environment)
        # high turning point 5*8, low turning point 5*8, technical indicator 4*8
        self.model = Model(2, 112)
        self.__buy_order_agent = buy_order_agent
        self.state = None # save the state to be trained
        self.buy_action = None # save the action needed to pass to fit method
        self.__no_train = no_train
        self.__iteration = 0

    def get_buy_order_agent(self):
        return self.__buy_order_agent

    def set_buy_order_agent(self, buy_order_agent):
        self.__buy_order_agent = buy_order_agent

    def process_next_state(self, is_first=False):
        self.state, self.buy_action = self.produce_state_and_get_action(is_first)
        self.process_action(self.buy_action)

    def process_action(self, buy_action):
        if not buy_action:
            self.process_next_state()
        else:
            self.invoke_buy_order_agent()

    def update_reward(self, from_but_order_agent, bp=None, sp=None):
        if from_but_order_agent:
            reward = 0
            self.model.fit(self.state, reward, self.buy_action)

        else:
            reward = ((1 - self.environment.transaction_cost) * sp - bp) / bp
            self.model.fit(self.state, reward, self.buy_action)
            self.__iteration = self.__iteration + 1
            if self.__iteration < self.__no_train:
                self.start_new_training()

    def invoke_buy_order_agent(self):
        self.__buy_order_agent.start_new_training()

    def start_new_training(self):
        print("Buy signal - start new training")
        self.process_next_state(is_first=True)
