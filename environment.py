import random

import data_engineering
import pandas as pd

from buy_order_agent import BuyOrderAgent
from buy_signal_agent import BuySignalAgent
from sell_order_agent import SellOrderAgent
from sell_signal_agent import SellSignalAgent


class Environment:
    def __init__(self):
        self.data = None
        self.states = None
        self.transaction_cost = 1
        self.turning_point_max, self.turning_point_min = data_engineering.create_turning_point_matrix(self.data)

    @staticmethod
    def get_random_action(agent):
        if isinstance(agent, BuyOrderAgent):
            action = random.randint(0, 7)
        elif isinstance(agent, SellOrderAgent):
            action = random.randint(0, 7)
        elif isinstance(agent, BuySignalAgent):
            action = random.randint(0, 2)
        elif isinstance(agent, SellSignalAgent):
            action = random.randint(0, 2)
        return action

    def generate_sell_signal_states(self, last_date):
        # get next day state, if next day state is not available, throws error
        s = "generated sellSignalStates"
        return s

    def generate_buy_signal_states(self, last_date):
        if last_date == None:
            # randomly pick a day from dataset
            s = "generated buySignalStates - first"
        else:
            s = "generated buySignalStates"
        return s

    def generate_sell_order_states(self, last_date):
        # get next day state, if next day state is not available, throws error
        s = "generated sellOrderStates"
        return s

    def generate_buy_order_states(self, last_date):
        # get next day state, if next day state is not available, throws error
        s = "generated buyOrderStates"
        return s

    def load_data(self):
        # load from csv/ db
        pass

    def get_market_data_by_date_of_state(self, state):
        sample_df = pd.DataFrame(data={'ma5': [1], 'low': [3]})
        return sample_df

    def produce_state(self, agent, last_date):
        if isinstance(agent, BuyOrderAgent):
            s = self.generate_buy_order_states(last_date)
        elif isinstance(agent, SellOrderAgent):
            s = self.generate_sell_order_states(last_date)
        elif isinstance(agent, BuySignalAgent):
            s = self.generate_buy_signal_states(last_date)
        elif isinstance(agent, SellSignalAgent):
            s = self.generate_sell_signal_states(last_date)

        print(s)
        return s
