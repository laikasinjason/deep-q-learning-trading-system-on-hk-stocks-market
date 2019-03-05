import random

import pandas as pd

import data_engeering.data_engineering as data_engineering
from buy_order_agent import BuyOrderAgent
from buy_signal_agent import BuySignalAgent
from sell_order_agent import SellOrderAgent
from sell_signal_agent import SellSignalAgent


class Environment:
    def __init__(self):
        self.data = None
        self.transaction_cost = 1
        data_engineering.enrich_market_data(self.data)
        self.turning_point_max, self.turning_point_min = data_engineering.create_turning_point_4d_matrix(self.data)
        self.tech_indicator_matrix = data_engineering.create_technical_indicator_4d_matrix(self.data)

    @staticmethod
    def get_random_action(agent):
        global action
        if isinstance(agent, BuyOrderAgent):
            action = random.randint(0, 7)
        elif isinstance(agent, SellOrderAgent):
            action = random.randint(0, 7)
        elif isinstance(agent, BuySignalAgent):
            action = random.randint(0, 2)
        elif isinstance(agent, SellSignalAgent):
            action = random.randint(0, 2)
        return action

    def get_sell_signal_states_by_date(self, bp, date):
        # get next day state, if next day state is not available, throws error
        print("generated sellSignalStates")
        
        try:
            td_market_data = self.data.loc[date]['close']
            profit = ( td_market_data - bp  ) / bp
            
            temp_series = pd.Series([profit])
            profit_bin = pd.get_dummies(pd.cut(temp_series, data_engineering.bins, labels=data_engineering.names))
            
            tp_max = self.turning_point_max.loc[date]
            tp_min = self.turning_point_max.loc[date]
            tech_indicator = self.tech_indicator_matrix.loc[date]
            s = np.concatenate((tp_max.values.flatten(), tp_min.values.flatten(), tech_indicator.values.flatten(), profit_bin.values.flatten()), axis = 0)
            return s
        except KeyError:
            print("ERROR getting sell signal state")
            return None

    def get_buy_signal_states_by_date(self, date):
        if date is None:
            # randomly pick a day from dataset
            date = random.randint(1,1)
            print("generated buySignalStates - first")
        else:
            print("generated buySignalStates")
                
        try:
            tp_max = self.turning_point_max.loc[date]
            tp_min = self.turning_point_max.loc[date]
            tech_indicator = self.tech_indicator_matrix.loc[date]
            s = np.concatenate((tp_max.values.flatten(), tp_min.values.flatten(), tech_indicator.values.flatten()), axis = 0)
            return s
        except KeyError:
            print("ERROR getting buy signal state")
            return None
 

    def get_sell_order_states_by_date(self, date):
        # get next day state, if next day state is not available, throws error
        print("generated sellOrderStates")
        
        try:
            tech_indicator = self.tech_indicator_matrix.loc[date]
            s = tech_indicator.values.flatten()
            return s
        except KeyError:
            print("ERROR getting sell order state")
            return None
 
    def get_buy_order_states_by_date(self, date):
        # get next day state, if next day state is not available, throws error
        print("generated buyOrderStates")

        try:
            tech_indicator = self.tech_indicator_matrix.loc[date]
            s = tech_indicator.values.flatten()
            return s
        except KeyError:
            print("ERROR getting buy order state")
            return None

    def load_data(self):
        # load from csv/ db
        pass

    def get_market_data_by_date_of_state(self, date):
        return self.data.loc[date]

    def produce_state(self, agent, last_date):
        if isinstance(agent, BuyOrderAgent):
            s = self.get_buy_order_states_by_date(last_date+1)
        elif isinstance(agent, SellOrderAgent):
            s = self.get_sell_order_states_by_date(last_date+1)
        elif isinstance(agent, BuySignalAgent):
            s = self.get_buy_signal_states_by_date(last_date+1)
        elif isinstance(agent, SellSignalAgent):
            print("buy price: " + agent.bp)
            s = self.get_sell_signal_states_by_date(agent.bp, last_date+1)

        print(s)
        return s
