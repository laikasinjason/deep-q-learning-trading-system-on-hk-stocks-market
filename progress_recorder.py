class ProgressRecorder():
    # Class for saving the performance indicators

    def __init__(self):
        self.cum_profit = 0
        self.max_cum_profit = 0
        self.drawdown = 0

    def reset(self):
        self.cum_profit = 0
        self.max_cum_profit = 0
        self.drawdown = 0

    def process_recorded_data(self, **data):
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

        self.cum_profit = self.cum_profit + profit

        self.max_cum_profit = self.cum_profit if self.cum_profit > self.max_cum_profit else self.max_cum_profit
        self.drawdown = self.max_cum_profit - self.cum_profit if self.max_cum_profit > self.cum_profit else 0

        result = str(date) + "," + str(bp) + "," + str(sp) + "," + str(profit) + "," + str(self.cum_profit) + "," + \
                 str(self.drawdown) + "\n"

        self.write_to_file(result)

    @staticmethod
    def write_to_file(line):
        print("Writing line: " + line)
        f = open("progress.txt", "a")
        f.write(line)
