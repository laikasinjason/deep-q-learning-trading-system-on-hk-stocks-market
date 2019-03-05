import random

import numpy as np
import pandas as pd

import data_engineering.data_engineering as data_engineering
from buy_order_agent import BuyOrderAgent
from buy_signal_agent import BuySignalAgent
from sell_order_agent import SellOrderAgent
from sell_signal_agent import SellSignalAgent


class Environment:
    class State:
        def __init__(self, date, value):
            self.date = date
            self.value = value

    def __init__(self):
        self.data = None
        self.transaction_cost = 1
        data_engineering.enrich_market_data(self.data)

        self.turning_point_max, self.turning_point_min = data_engineering.create_turning_point_4d_matrix(self.data)
        self.tech_indicator_matrix = data_engineering.create_technical_indicator_4d_matrix(self.data)

        # simulate data for testing
        self.data = pd.DataFrame(
            {'date': [i for i in range(100)], 'low': [2 * i for i in range(100)], 'high': [7 * i for i in range(100)],
             'ma5': [3 * i for i in range(100)], 'close': [5 * i for i in range(100)]}).set_index('date')
        self.turning_point_max = pd.DataFrame(
            {'date': [i for i in range(100)], 'col2': [2 * i for i in range(100)]}).set_index('date')
        self.turning_point_min = pd.DataFrame(
            {'date': [i for i in range(100)], 'col2': [2 * i for i in range(100)]}).set_index('date')
        self.tech_indicator_matrix = pd.DataFrame(
            {'date': [i for i in range(100)], 'col2': [2 * i for i in range(100)]}).set_index('date')

    @staticmethod
    def get_random_action(agent):
        global action
        if isinstance(agent, BuyOrderAgent):
            action = random.randint(0, 7)
        elif isinstance(agent, SellOrderAgent):
            action = random.randint(0, 7)
        elif isinstance(agent, BuySignalAgent):
            action = random.randint(0, 1)
        elif isinstance(agent, SellSignalAgent):
            action = random.randint(0, 1)
        return action

    def get_sell_signal_states_by_date(self, bp, date):
        # get next day state, if next day state is not available, throws error
        try:
            td_market_data = self.data.loc[date]['close']
            profit = (td_market_data - bp) / bp

            temp_series = pd.Series([profit])
            profit_bin = pd.get_dummies(pd.cut(temp_series, data_engineering.bins, labels=data_engineering.names))

            tp_max = self.turning_point_max.loc[date]
            tp_min = self.turning_point_max.loc[date]
            tech_indicator = self.tech_indicator_matrix.loc[date]
            s = np.concatenate((tp_max.values.flatten(), tp_min.values.flatten(), tech_indicator.values.flatten(),
                                profit_bin.values.flatten()), axis=0)
            state = self.State(date, s)
            print("generated sellSignalStates, date " + str(date))
            return state
        except KeyError:
            print("ERROR getting sell signal state for date " + str(date))
            return None

    def get_buy_signal_states_by_date(self, date):
        try:
            tp_max = self.turning_point_max.loc[date]
            tp_min = self.turning_point_max.loc[date]
            tech_indicator = self.tech_indicator_matrix.loc[date]
            s = np.concatenate((tp_max.values.flatten(), tp_min.values.flatten(), tech_indicator.values.flatten()),
                               axis=0)
            state = self.State(date, s)
            print("generated buySignalStates, date " + str(date))
            return state
        except KeyError:
            print("ERROR getting buy signal state for date " + str(date))
            return None

    def get_sell_order_states_by_date(self, date):
        # get next day state, if next day state is not available, throws error

        try:
            tech_indicator = self.tech_indicator_matrix.loc[date]
            s = tech_indicator.values.flatten()
            state = self.State(date, s)
            print("generated sellOrderStates, date " + str(date))
            return state
        except KeyError:
            print("ERROR getting sell order state for date " + str(date))
            return None

    def get_buy_order_states_by_date(self, date):
        # get next day state, if next day state is not available, throws error

        try:
            tech_indicator = self.tech_indicator_matrix.loc[date]
            s = tech_indicator.values.flatten()
            state = self.State(date, s)
            print("generated buyOrderStates, date " + str(date))
            return state
        except KeyError:
            print("ERROR getting buy order state for date " + str(date))
            return None

    def load_data(self):
        # load from csv/ db
        pass

    def get_market_data_by_date(self, date):
        market_data = self.data.loc[date]
        print("Getting market data, date: " + str(date) + " , " + str(market_data))

        return market_data

    def produce_state(self, agent, last_date):
        if isinstance(agent, BuyOrderAgent):
            s = self.get_buy_order_states_by_date(last_date + 1)
        elif isinstance(agent, SellOrderAgent):
            s = self.get_sell_order_states_by_date(last_date + 1)
        elif isinstance(agent, BuySignalAgent):
            if last_date is None:
                # randomly pick a day from dataset
                # date =98
                date = self.data.sample().index.values[0]
                print("generated buySignalStates - first")
                s = self.get_buy_signal_states_by_date(date)
            else:
                s = self.get_buy_signal_states_by_date(last_date + 1)
        elif isinstance(agent, SellSignalAgent):
            s = self.get_sell_signal_states_by_date(agent.bp, last_date + 1)

        if s is not None:
            print(str(s.date) + " , " + str(s.value))
        return s
