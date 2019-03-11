import math

from agent import Agent
from model import Model


class SellOrderAgent(Agent):
    def __init__(self, environment, buy_signal_agent=None):
        super().__init__(environment)

        self.bp = None
        # technical indicator 4*8
        self.model = Model(2, 32)
        self.__buy_signal_agent = buy_signal_agent
        self.state = None  # save the state to be trained
        self.action = None  # save the action needed to pass to fit method

    def get_buy_signal_agent(self):
        return self.__buy_signal_agent

    def set_buy_signal_agent(self, buy_signal_agent):
        self.__buy_signal_agent = buy_signal_agent

    def process_action(self, action, date):
        # sell order agent consider state on T-1, and place order on T day

        next_date = self.environment.get_next_day_of_state(date)
        if next_date is None:
        # not able to get next date's market data
            return True
        
        market_data = self.environment.get_market_data_by_date(next_date)


        if market_data is None:
            # terminated
            return True

        ma5 = market_data['ma5']
        high = market_data['high']

        if ma5 is None or high is None:
            # terminate
            return True

        d = ma5 + action / 100 * ma5 - high
        print("processing sell order")

        if d <= 0:
            reward = math.exp(100 * d / high)
            sp = ma5 + action / 100 * ma5
            # self.model.fit(self.state, reward, action)
            print("sell price: " + str(sp))
            self.invoke_buy_signal_agent(sp)

        else:
            reward = 0
            # self.model.fit(self.state, reward, action)

            close = 3
            sp = close
            self.invoke_buy_signal_agent(sp)
        return False

    def invoke_buy_signal_agent(self, sp):
        self.__buy_signal_agent.update_reward(False, self.bp, sp)

    def restart_training(self, terminated_by_other_agents=True):
        # state is not available, restart from the top
        self.__buy_signal_agent.start_new_training(terminated_by_other_agents)

    def start_new_training(self, bp, date):
        print("Sell order - start new training " + str(date))
        self.bp = bp
        state = self.environment.get_sell_order_states_by_date(date)
        action = self.get_action(state)

        terminated = self.process_action(action, date)

        if terminated:
            self.restart_training()
