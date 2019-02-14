from agent import Agent
from environment import transaction_cost
from model import Model


class BuySignalAgent(Agent):
    def __init__(self, environment, no_train, buy_order_agent=None):
		super().__init__(environment)
        self.model = Model(2, 50)
        self.__buy_order_agent = buy_order_agent
		self.state = None
		self.__no_train = no_train
		self.__iteration = 0

    def get_buy_order_agent(self):
        return self.__buy_order_agent

    def set_buy_order_agent(self, buy_order_agent):
        self.__buy_order_agent = buy_order_agent

    def process_next_state(self, isFirst=False):
        self.state, buy_action = self.produce_state_and_get_action(isFirst)
        self.process_action(buy_action)

    def process_action(self, buy_action):
        if not buy_action:
            self.process_next_state()
        else:
            self.invoke_buy_order_agent()

    def update_reward(self, from_but_order_agent, bp=None, sp=None):
        if from_but_order_agent:
            reward = 0
			self.model.fit(self.state, reward)
        else:
            reward = ((1 - transaction_cost) * sp - bp) / bp
			self.model.fit(self.state, reward)
		self.__iteration = self.__iteration + 1
		if (self.__iteration < self.__no_train):
			self.start_new_training()

    def invoke_buy_order_agent(self):
        self.__buy_order_agent.start_new_training()
		
	def start_new_training(self):
        print("Buy signal - start new training")
        self.process_next_state(isFirst=True)
