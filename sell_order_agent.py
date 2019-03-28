import math

from agent import Agent
from model import OrderModel


class SellOrderAgent(Agent):
    def __init__(self, environment):
        super().__init__(environment)

        # technical indicator 4*8
        self.model = OrderModel(7, 32)
        self.state = None  # save the state to be trained
        self.action = None  # save the action needed to pass to fit method

    def process_action(self, action, date):
        # sell order agent consider state on T-1, and place order on T day
        market_data = self.environment.get_market_data_by_date(date)

        if market_data is None:
            # terminated
            return False

        ma5 = market_data['ma5']
        high = market_data['High']
        close = market_data['Close']

        if ma5 is None or high is None:
            # terminate
            return False

        sp = ma5 + action / 100 * ma5
        d = sp - high

        if d <= 0:
            reward = math.exp(100 * d / high)

        else:
            reward = 0
            sp = close

        if not self.environment.get_evaluation_mode():
            self.model.fit(self.state.value, reward, action)
        else:
            profit = (1 - self.environment.transaction_cost) * sp - self.environment.get_buy_price()
            pf_return = (1 - self.environment.transaction_cost) * sp / self.environment.get_buy_price()
            record = {'sp': sp, 'date': date, 'profit': profit, 'return': pf_return}
            self.environment.record(**record)
        print("processing sell order, sell price: " + str(sp))
        self.environment.invoke_buy_signal_agent(sp, date, self.environment.get_buy_price(), sp)
        return True

    def process_next_state(self, date):
        # the date get here is already the next day, but we need the same day of SSA as the state
        prev_date = self.environment.get_prev_day(date)
        print("Sell order - processing date: " + str(date))
        self.state = self.environment.get_sell_order_states_by_date(prev_date)
        action = self.get_action(self.state)

        result = self.process_action(action, date)
        if not result:
            self.environment.process_epoch_end(None, True)
