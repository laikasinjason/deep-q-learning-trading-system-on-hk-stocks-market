from agent import Agent
from model import Model


class SellSignalAgent(Agent):
    def __init__(self, environment, sell_order_agent=None):
        super().__init__(environment)

        self.bp = None
        # technical indicator 4*8
        self.model = Model(2, 50)
        self.__sell_order_agent = sell_order_agent
        self.state = None # save the state to be trained
        self.action = None # save the action needed to pass to fit method

    def get_sell_order_agent(self):
        return self.__sell_order_agent

    def set_sell_order_agent(self, sell_order_agent):
        self.__sell_order_agent = sell_order_agent

    def process_action(self, sell_action):
        market_data = self.environment.get_market_data_by_date_of_state(self.state)
        if not sell_action:
            # force sell signal agent to sell if profit is in certain condition
            if((market_data['profit'] > 0.3) or (market_data['profit'] < -0.2)):
                self.invoke_sell_order_agent()
             else:
                reward = market_data['rc']
                self.model.fit(self.state, reward, sell_action)
                self.process_next_state()
        else:
            self.invoke_sell_order_agent()

    def process_next_state(self):
        self.state, sell_action = self.produce_state_and_get_action()
        self.process_action(sell_action)

    def invoke_sell_order_agent(self):
        self.__sell_order_agent.start_new_training(self.bp)

    def start_new_training(self, bp):
        print("Sell signal - start new training")
        self.bp = bp
        self.process_next_state()
