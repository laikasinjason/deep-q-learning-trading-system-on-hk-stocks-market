class ProgressRecorder():
    # Class for saving the performance indicators

    def __init__(self):
        self.cumProfit = 0

    def reset(self):
        self.cumProfit = 0

    def evaluate(self, env):

        print("Evaluation started.")

        env.set_evaluation_mode(True)

        result = env.start_new_epoch()
        while result is not None:
            # able to get next date's market data, continue to trade in evaluation mode
            print("Progress: Next date: " + str(result))
            result = env.start_new_epoch()
        env.set_evaluation_mode(False)

    def process_recorded_data(self, **data):
        # date,bp,sp,profit,cumProfit
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

        self.cumProfit = self.cumProfit + profit

        result = str(date) + "," + str(bp) + "," + str(sp) + "," + str(profit) + "," + str(self.cumProfit) + "\n"
        self.progress_recorder.write_to_file(result)

    @staticmethod
    def write_to_file(line):
        print("Writing line: " + line)
        f = open("progress.txt", "a")
        f.write(line)
