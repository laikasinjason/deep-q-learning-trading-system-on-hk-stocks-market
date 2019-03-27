class ProgressRecorder():
    # Class for saving the performance indicators

    def __init__(self):
        self.__cum_profit = 0
        self.__max_cum_profit = 0
        self.__drawdown = 0
        self.__max_drawdown = 0

    def reset(self):
        self.___cum_profit = 0
        self.max__cum_profit = 0
        self.__drawdown = 0
        self.__max_drawdown = 0

    def process_recorded_data(self, **data, write_to_file=False):
        # date,bp,sp,profit,cumProfit,drawdown
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

        self.__cum_profit = self.__cum_profit + profit

        self.__max_cum_profit = self.__cum_profit if self.__cum_profit > self.__max_cum_profit else self.__max_cum_profit
        self.__drawdown = self.__max_cum_profit - self.__cum_profit if self.__max_cum_profit > self.__cum_profit else 0
        self.__max_drawdown = self.__drawdown if self.__drawdown > self.__max_drawdown else self.__max_drawdown

        if write_to_file:
            result = str(date) + "," + str(bp) + "," + str(sp) + "," + str(profit) + "," + str(self.__cum_profit) + "," + \
                     str(self.__drawdown) + "\n"

            self.write_to_file(result, "evaluation.txt")

    def get_max_drawdown(self):
        return self.__max_drawdown
        
    def get_cum_profit(self):
        return self.__cum_profit
        
    def write_after_evaluation_end(self, iteration):
        result = str(iteration) + "," +  str(self.__cum_profit) + "," + str(self.__max_drawdown) + "\n"

        self.write_to_file(result, "training_progress.txt")
        
        print("Iteration: " + self.__iteration + "/" + self.__num_train + ", "\
                + "evaluation: max drawdown, cum profit " + self.progress_recorder.get_max_drawdown() + ", "\
                + self.progress_recorder.get_cum_profit())
        
    @staticmethod
    def write_to_file(line, file):
        print("Writing line: " + line)
        f = open(file, "a")
        f.write(line)
