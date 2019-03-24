from agent import Agent
from model import SellSignalModel


class SellSignalAgent(Agent):
    def __init__(self, environment, sell_order_agent=None):
        super().__init__(environment)

        # high turning point 5*8, low turning point 5*8, technical indicator 4*8, profit 8
        self.model = SellSignalModel(2, 120)
        self.state = None  # save the state to be trained
        self.action = None  # save the action needed to pass to fit method

    def process_action(self, sell_action, last_state_date):
        market_data = self.environment.get_market_data_by_date(last_state_date)
        # get next day for training
        next_day = self.environment.get_next_day(last_state_date)

        if (market_data is None) or (next_day is None):
            # terminated
            return False

        # for training
        next_state = self.environment.get_sell_signal_states_by_date(self.environment.get_buy_price(), next_day)

        close = market_data['Close']
        roc = market_data['rate_of_close']

        if close is None:
            # terminate
            return False

        profit = (self.bp - close) / close

        if sell_action or (profit > 0.3) or (profit < -0.2):
            # force sell signal agent to sell if profit is in certain condition, or sell action
            self.environment.invoke_sell_order_agent()
        else:
            reward = roc
            # if not self.environment.get_evaluation_mode():
            # self.model.fit(self.state.value, reward, sell_action, next_state)

        return True

    def process_next_state(self, date):
        print("Sell signal - processing date: " + str(date))

        self.state, sell_action = self.produce_state_and_get_action(date)
        if self.state is None or sell_action is None:
            # stop training
            self.environment.process_epoch_end(None, True)
        else:
            this_state_date = self.state.date
            result = self.process_action(sell_action, this_state_date)
            if not result:
                self.environment.process_epoch_end(None, True)
