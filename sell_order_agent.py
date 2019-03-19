import math

from agent import Agent
from model import OrderModel


class SellOrderAgent(Agent):
    def __init__(self, environment, buy_signal_agent=None):
        super().__init__(environment)

        self.bp = None
        # technical indicator 4*8
        self.model = OrderModel(7, 32)
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
            return False

        market_data = self.environment.get_market_data_by_date(next_date)

        if market_data is None:
            # terminated
            return False

        ma5 = market_data['ma5']
        high = market_data['High']

        if ma5 is None or high is None:
            # terminate
            return False

        sp = ma5 + action / 100 * ma5
        d = sp - high
        print("processing sell order, sell price: " + str(sp))

        if d <= 0:
            reward = math.exp(100 * d / high)

            # if not self.environment.get_evaluation_mode():
            # self.model.fit(self.state.value, reward, action)
            # else:
            # profit = (1 - self.environment.transaction_cost) * sp - bp
            # record = {'sp' : sp, 'date' : next_date, 'profit', profit}
            # self.environment.record(record)
            self.invoke_buy_signal_agent(sp, next_date)

        else:
            reward = 0
            # if not self.environment.get_evaluation_mode():
            # self.model.fit(self.state.value, reward, action)

            close = 3
            sp = close
            self.invoke_buy_signal_agent(sp, next_date)
        return True

    def invoke_buy_signal_agent(self, sp, date):
        self.__buy_signal_agent.update_reward(False, date, self.bp, sp)

    def start_new_training(self, bp, date):
        print("Sell order - start new training " + str(date))
        self.bp = bp
        self.state  = self.environment.get_sell_order_states_by_date(date)
        action = self.get_action(self.state)

        result = self.process_action(action, date)
        if not result:
            self.environment.terminate_epoch()
