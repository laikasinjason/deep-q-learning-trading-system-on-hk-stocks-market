import gc
import numpy as np
import pandas as pd

import data_engineering.data_engineering as data_engineering
import tensorflow as tf
import model_loading
from buy_order_agent import BuyOrderAgent
from buy_signal_agent import BuySignalAgent
from sell_order_agent import SellOrderAgent
from sell_signal_agent import SellSignalAgent


class Environment:

    class State:
        def __init__(self, date, value):
            self.date = date
            self.value = value
          
    __buy_signal_agent = None
    __sell_signal_agent = None
    __buy_order_agent = None
    __sell_order_agent = None
    # env variable
    __evaluation_mode = False
    __iteration = 0
    __terminated = None
    __date = None  # current date on training
    __bp = None
    __running_agent = None  # the active agent in the trading process
    max_tau = 1000 #Tau is the C step where we update our target network
        
    def __init__(self, csv_file, progress_recorder, num_train, transaction_cost=0.01):
        self.data = data_engineering.load_data(csv_file)
        self.transaction_cost = transaction_cost

        self.data = data_engineering.clean_data(self.data)
        self.turning_point_max, self.turning_point_min = data_engineering.create_turning_point_3d_matrix(self.data)
        self.tech_indicator_matrix = data_engineering.create_technical_indicator_3d_matrix(self.data)
        # add new cols, and truncate data so same row as above matrices
        self.data = data_engineering.enrich_market_data(self.data)

        # split data to train, test
        self.train_index, self.test_index = data_engineering.split_data_set_index(self.data)

        self.progress_recorder = progress_recorder

        # env variable
        self.__num_train = num_train

        # simulate data for testing
        # test_len = 5
        # self.data = pd.DataFrame(
        #     {'date': [i for i in range(test_len)], 'Low': [2 * i for i in range(test_len)],
        #      'High': [7 * i for i in range(test_len)], 'rate_of_close': [2 * i for i in range(test_len)],
        #      'ma5': [3 * i for i in range(test_len)], 'Close': [5 * i for i in range(test_len)]}).set_index('date')
        # self.turning_point_max = pd.DataFrame(
        #     {'date': [i for i in range(test_len)], 'col2': [2 * i for i in range(test_len)]}).set_index('date')
        # self.turning_point_min = pd.DataFrame(
        #     {'date': [i for i in range(test_len)], 'col2': [2 * i for i in range(test_len)]}).set_index('date')
        # self.tech_indicator_matrix = pd.DataFrame(
        #     {'date': [i for i in range(test_len)], 'col2': [2 * i for i in range(test_len)]}).set_index('date')

        self.assert_data_consistency()
        
        # reset tensorflow graph
        tf.reset_default_graph()

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
            # print("generated sellSignalStates, date " + str(date))
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
            # print("generated buySignalStates, date " + str(date))
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
            # print("generated sellOrderStates, date " + str(date))
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
            # print("generated buyOrderStates, date " + str(date))
            return state
        except KeyError:
            print("ERROR getting buy order state for date " + str(date))
            return None

    def get_market_data_by_date(self, date):
        market_data = self.data.loc[date]
        # print("Getting market data, date: " + str(date) + " , \n" + str(market_data))

        return market_data

    def produce_state(self, agent, date):
        s = None
        if isinstance(agent, BuyOrderAgent):
            s = self.get_buy_order_states_by_date(date)
        elif isinstance(agent, SellOrderAgent):
            s = self.get_sell_order_states_by_date(date)
        elif isinstance(agent, SellSignalAgent):
            s = self.get_sell_signal_states_by_date(self.__bp, date)
        elif isinstance(agent, BuySignalAgent):
            s = self.get_buy_signal_states_by_date(date)

        return s

    def get_evaluation_mode(self):
        return self.__evaluation_mode

    def get_buy_signal_agent(self):
        return self.__buy_signal_agent

    def set_agents(self, buy_signal_agent, sell_signal_agent, buy_order_agent, sell_order_agent):
        self.__buy_signal_agent = buy_signal_agent
        self.__sell_signal_agent = sell_signal_agent
        self.__buy_order_agent = buy_order_agent
        self.__sell_order_agent = sell_order_agent

    def record(self, **data):
        self.progress_recorder.process_recorded_data(**data)

    def evaluate(self, evaluation_write_file=False):
        print("Evaluation started.")
        self.progress_recorder.reset(evaluation_write_file)
        self.__evaluation_mode = True
        self.__date = self.test_index[0]

        while self.__evaluation_mode:
            # able to get next date's market data, continue to trade in evaluation mode
            self.start_new_epoch()
            gc.collect()

    def start_new_epoch(self):
        # from buy (open position) to sell(close position) is considered an epoch
        self.__running_agent = self.__buy_signal_agent
        self.__bp = None
        self.__terminated = False

        if not self.__evaluation_mode:
            self.__date = pd.Series(self.train_index).sample().values[0]

        while not self.__terminated:
            self.__running_agent.process_next_state(self.__date)

            if not self.__terminated:
                self.__date = self.get_next_day(self.__date)
                if self.__date is None:
                    self.process_epoch_end(None, True)
                    
    def fill_up_memory(self):
        while (!self.__buy_signal_agent.model.memory.is_full() or
            !self.__sell_signal_agent.model.memory.is_full() or
            !self.__buy_order_agent.model.memory.is_full() or
            !self.__sell_order_agent.model.memory.is_full()):
            self.start_new_epoch()

            gc.collect()
        
    def get_random_action(self):
        return self.__random_action

    def set_buy_price(self, bp):
        self.__bp = bp

    def get_buy_price(self):
        return self.__bp

    def get_iteration(self):
        return self.__iteration

    def set_iteration(self, iteration):
        self.__iteration = iteration

    def invoke_buy_order_agent(self):
        # invoking buy order agent with the state of the stock at the same day
        self.__running_agent = self.__buy_order_agent

    def invoke_sell_order_agent(self):
        self.__running_agent = self.__sell_order_agent

    def invoke_sell_signal_agent(self):
        self.__running_agent = self.__sell_signal_agent

    def invoke_buy_signal_agent(self, from_buy_order_agent, date, bp=None, sp=None):
        # when invoking BSA, it is to update the BSA's rewards
        self.__buy_signal_agent.update_reward(from_buy_order_agent, date, bp, sp)

    def process_epoch_end(self, end_date, terminate=False):
        if terminate:
            # reset evaluation mode if it is terminated in evaluation mode
            if self.__evaluation_mode:
                print("Terminated in evaluation mode")
                self.__evaluation_mode = False
            else:
                print("Terminated, iteration : " + str(self.__iteration))
                self.__date = None

        else:
            if self.__evaluation_mode:
                next_date_for_evaluation = self.get_next_day(end_date)
                if next_date_for_evaluation is None:
                    self.__evaluation_mode = False
            else:
                self.__iteration = self.__iteration + 1
                self.__date = None
                # print("iteration: " + str(self.__iteration) + "/" + str(self.__num_train))

        self.__terminated = True

    def get_next_day(self, date):
        return data_engineering.get_next_day(date, self.data)

    def get_prev_day(self, date):
        return data_engineering.get_prev_day(date, self.data)

    def train_system(self, num_train=None):
        if num_train is not None:
            self.__num_train = num_train

        while self.__iteration < self.__num_train:
            self.start_new_epoch()

            if self.__iteration % max_tau == 0:  # 1000
                # self.__sell_signal_agent.model.save_target_model()
                self.__sell_signal_agent.model.update_target_graph()
                
                print("Saved sell signal agent's target model.")
            if self.__iteration % 10000 == 0:  # 10000
                self.evaluate()
                self.progress_recorder.write_after_evaluation_end(self.__iteration, self.__num_train)
                model_loading.save_tf_model(self.__buy_signal_agent)
                model_loading.save_tf_model(self.__buy_order_agent)
                model_loading.save_tf_model(self.__sell_signal_agent)
                model_loading.save_tf_model(self.__sell_order_agent)

            gc.collect()

    def load_model(self):
        model_loading.load_model(self.__buy_signal_agent)
        model_loading.load_model(self.__buy_order_agent)
        model_loading.load_model(self.__sell_signal_agent)
        model_loading.load_model(self.__sell_order_agent)

    def assert_data_consistency(self):
        assert len(self.data) == len(self.turning_point_max.index.levels[0])
        assert len(self.turning_point_max.index.levels[0]) == len(self.turning_point_min.index.levels[0])
        assert len(self.turning_point_min.index.levels[0]) == len(self.tech_indicator_matrix.index.levels[0])
