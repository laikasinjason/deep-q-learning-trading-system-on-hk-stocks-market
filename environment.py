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
        enrich_market_data(self.data)
        self.turning_point_max, self.turning_point_min = data_engineering.create_turning_point_4dmatrix(self.data)
        self.order_agent_matrix = data_engineering.create_order_agent_4dmatrix(self.data)

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

    def get_sell_signal_states_by_date(self, last_date):
        # get next day state, if next day state is not available, throws error
        s = "generated sellSignalStates"
        return s

    def get_buy_signal_states_by_date(self, last_date):
        if last_date == None:
            # randomly pick a day from dataset
            s = "generated buySignalStates - first"
        else:
            s = "generated buySignalStates"
        return s

    def get_sell_order_states_by_date(self, last_date):
        # get next day state, if next day state is not available, throws error
        s = "generated sellOrderStates"
        
        # s = self.order_agent_matrix['last_date']
        return s

    def get_buy_order_states_by_date(self, last_date):
        # get next day state, if next day state is not available, throws error
        s = "generated buyOrderStates"
        
        # s = self.order_agent_matrix['last_date']
        return s

    def load_data(self):
        # load from csv/ db
        pass

    def get_market_data_by_date_of_state(self, date):
        return self.data[date]

    def produce_state(self, agent, last_date):
        if isinstance(agent, BuyOrderAgent):
            s = self.get_buy_order_states_by_date(last_date)
        elif isinstance(agent, SellOrderAgent):
            s = self.get_sell_order_states_by_date(last_date)
        elif isinstance(agent, BuySignalAgent):
            s = self.get_buy_signal_states_by_date(last_date)
        elif isinstance(agent, SellSignalAgent):
            s = self.get_sell_signal_states_by_date(last_date)

        print(s)
        return s
