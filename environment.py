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

    def __init__(self, csv_file, progress_recorder, transaction_cost=0.01):
        self.data = data_engineering.load_data(csv_file)
        self.transaction_cost = transaction_cost
        data_engineering.enrich_market_data(self.data)

        self.turning_point_max, self.turning_point_min = data_engineering.create_turning_point_3d_matrix(self.data)
        self.tech_indicator_matrix = data_engineering.create_technical_indicator_3d_matrix(self.data)
        
        self.__evaluation_mode = False
        self.progress_recorder = progress_recorder

        # simulate data for testing
        test_len = 5
        self.data = pd.DataFrame(
            {'date': [i for i in range(test_len)], 'Low': [2 * i for i in range(test_len)],
             'High': [7 * i for i in range(test_len)],
             'ma5': [3 * i for i in range(test_len)], 'Close': [5 * i for i in range(test_len)]}).set_index('date')
        self.turning_point_max = pd.DataFrame(
            {'date': [i for i in range(test_len)], 'col2': [2 * i for i in range(test_len)]}).set_index('date')
        self.turning_point_min = pd.DataFrame(
            {'date': [i for i in range(test_len)], 'col2': [2 * i for i in range(test_len)]}).set_index('date')
        self.tech_indicator_matrix = pd.DataFrame(
            {'date': [i for i in range(test_len)], 'col2': [2 * i for i in range(test_len)]}).set_index('date')

    def get_sell_signal_states_by_date(self, bp, date):
        # get next day state, if next day state is not available, throws error
        try:
            td_market_data = self.data.loc[date]['Close']
            profit = (td_market_data - bp) / bp

            temp_series = pd.Series([profit])
            profit_bin = pd.get_dummies(pd.cut(temp_series, data_engineering.bins, labels=data_engineering.names))
            if profit == 0.0:
                profit_bin[:] = 0

            tp_max = self.turning_point_max.loc[date]
            tp_min = self.turning_point_min.loc[date]
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
            tp_min = self.turning_point_min.loc[date]
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

    def get_next_day_of_state(self, date):
        next_day = data_engineering.get_next_day(date, self.data)
        if next_day is None:
            print("ERROR getting state for date " + str(date))
        return next_day

    def get_market_data_by_date(self, date):
        market_data = self.data.loc[date]
        print("Getting market data, date: " + str(date) + " , " + str(market_data))

        return market_data

    def produce_state(self, agent, last_date):
        if (last_date is None) and isinstance(agent, BuySignalAgent):
            # randomly pick a day from dataset
            date = self.data.sample().index.values[0]
            print("generated buySignalStates - first")
            s = self.get_buy_signal_states_by_date(date)

        else:
            next_day = self.get_next_day_of_state(last_date)
            if next_day is None:
                return None

            if isinstance(agent, BuyOrderAgent):
                s = self.get_buy_order_states_by_date(next_day)
            elif isinstance(agent, SellOrderAgent):
                s = self.get_sell_order_states_by_date(next_day)
            elif isinstance(agent, SellSignalAgent):
                s = self.get_sell_signal_states_by_date(agent.bp, next_day)
            elif isinstance(agent, BuySignalAgent):
                if last_date is None:
                    # randomly pick a day from dataset
                    date = self.data.sample().index.values[0]
                    print("generated buySignalStates - first")
                    s = self.get_buy_signal_states_by_date(date)
                else:
                    s = self.get_buy_signal_states_by_date(next_day)

        if s is not None:
            print("State: " + str(s.date) + " , " + str(s.value))
        return s
        
    def get_evaluation_mode(self):
        return self.__evaluation_mode
        
    def set_evaluation_mode(self, mode):
        self.__evaluation_mode = mode
        
    def record(**data):
        # date,bp,sp,profit
        date = data['date']
        bp = 0
        sp = 0
        profit = 0
        if 'bp' in data.keys():
            bp = data['bp']
        if 'sp' in data.keys():
            sp = data['sp']
        if 'profit' in data.keys():
            profit = data['profit']
            
        result = str(date)+","+str(bp)+","+str(sp)+","+str(profit)+"\n"
        self.progress_recorder.write_to_file(result)
