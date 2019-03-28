import pandas as pd

class ProgressRecorder():
    # Class for saving the performance indicators

    def __init__(self):
        self.__cum_profit = 0
        self.__max_cum_profit = 0
        self.__drawdown = 0
        self.__max_drawdown = 0
        self.__return_list = []
        self.__write_to_file = False

    def reset(self, evaluation_write_file):
        self.___cum_profit = 0
        self.max__cum_profit = 0
        self.__drawdown = 0
        self.__max_drawdown = 0
        self.__return_list = []
        self.__write_to_file = evaluation_write_file

    def process_recorded_data(self, **data):
        # date,bp,sp,profit,cumProfit,drawdown,return
        date = data['date']
        bp = 0
        sp = 0
        profit = 0
        pf_return = 0
        if 'bp' in data.keys():
            bp = data['bp']
        if 'sp' in data.keys():
            sp = data['sp']
        if 'profit' in data.keys():
            profit = data['profit']
        if 'return' in data.keys():
            pf_return = data['return']    

        self.__cum_profit = self.__cum_profit + profit
        self.__return_list.append(pf_return)

        self.__max_cum_profit = self.__cum_profit if self.__cum_profit > self.__max_cum_profit else self.__max_cum_profit
        self.__drawdown = self.__max_cum_profit - self.__cum_profit if self.__max_cum_profit > self.__cum_profit else 0
        self.__max_drawdown = self.__drawdown if self.__drawdown > self.__max_drawdown else self.__max_drawdown

        if self.__write_to_file:
            result = str(date) + "," + str(bp) + "," + str(sp) + "," + str(profit) + "," + str(self.__cum_profit) + "," + \
                     str(self.__drawdown) +  "," + str(pf_return) "\n"

            self.write_to_file(result, "evaluation.txt")

    def get_max_drawdown(self):
        return self.__max_drawdown
        
    def get_cum_profit(self):
        return self.__cum_profit
        
    def write_after_evaluation_end(self, iteration):
        # iteration,cumProfit,MaxDrawdown,SharpeRatio
        pd_returns = pd.Series(self.__return_list)
        sharpe_ratio = pd_returns.mean/pd_returns.std()
 
        result = str(iteration) + "," +  str(self.__cum_profit) + "," + str(self.__max_drawdown) + \
                 "," + str(sharpe_ratio) +"\n"

        self.write_to_file(result, "training_progress.txt")
        
        print("Iteration: " + self.__iteration + "/" + self.__num_train + ", "\
                + "evaluation: max drawdown, cum profit " + self.progress_recorder.get_max_drawdown() + ", "\
                + self.progress_recorder.get_cum_profit())
        
    @staticmethod
    def write_to_file(line, file):
        print("Writing line: " + line)
        f = open(file, "a")
        f.write(line)
