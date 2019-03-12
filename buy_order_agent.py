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
        # transform the action to a value in percentage
        action_value = self.model.action_map_to_value(action)
        
        # buy order agent consider state on T-1, and place order on T day

        next_date = self.environment.get_next_day_of_state(date)
        if next_date is None:
        # not able to get next date's market data
            return True
            
        market_data = self.environment.get_market_data_by_date(next_date)


        if market_data is None:
            # terminate
            return True

        ma5 = market_data['ma5']
        low = market_data['Low']

        if ma5 is None or low is None:
            # terminate
            return True

        d = ma5 + action_value / 100 * ma5 - low
        print("processing buy order")

        if d >= 0:
            reward = math.exp(-100 * d / low)
            bp = ma5 + action_value / 100 * ma5
            # self.model.fit(self.state.value, reward, action)
            # last state date for sell signal becomes T day, start training on T+1
            print("buy price: " + str(bp))
            self.invoke_sell_signal_agent(bp, next_date)
        else:
            reward = 0
            # self.model.fit(self.state.value, reward, action)
            self.invoke_buy_signal_agent()
        return False

    def invoke_sell_signal_agent(self, bp, last_state_date):
        self.__sell_signal_agent.start_new_training(bp, last_state_date)

    def invoke_buy_signal_agent(self):
        self.__buy_signal_agent.update_reward(True)

    def restart_training(self):
        # state is not available, restart from the top
        self.__buy_signal_agent.start_new_training(terminated_by_other_agents=True)

    def start_new_training(self, date):
        print("Buy order - start new training " + str(date))
        state = self.environment.get_buy_order_states_by_date(date)
        action = self.get_action(state)
        terminated = self.process_action(action, date)

        if terminated:
            self.restart_training()
