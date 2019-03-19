from agent import Agent
from model import SignalModel


class BuySignalAgent(Agent):
    def __init__(self, environment, buy_order_agent=None):
        super().__init__(environment)
        # high turning point 5*8, low turning point 5*8, technical indicator 4*8
        self.model = SignalModel(2, 112)
        self.__buy_order_agent = buy_order_agent
        self.state = None  # save the state to be trained
        self.buy_action = None  # save the action needed to pass to fit method

    def get_buy_order_agent(self):
        return self.__buy_order_agent

    def set_buy_order_agent(self, buy_order_agent):
        self.__buy_order_agent = buy_order_agent

    def process_next_state(self, last_state_date=None):
        self.state, self.buy_action = self.produce_state_and_get_action(last_state_date)

        if self.state is None or self.buy_action is None:
            self.environment.terminate_epoch()
        else:
            # get the date of this state
            this_state_date = self.state.date
            self.process_action(self.buy_action, this_state_date)

    def process_action(self, buy_action, last_state_date):
        if not buy_action:
            self.process_next_state(last_state_date)
        else:
            self.invoke_buy_order_agent()

    def update_reward(self, from_buy_order_agent, date, bp=None, sp=None):
        if not self.environment.get_evaluation_mode():
            if from_buy_order_agent:
                reward = 0
                print("reward: " + str(reward) + ", state: " + str(date) + ", bp: " + str(bp) + ", sp: " + str(sp))
                # self.model.fit(self.state.value, reward, self.buy_action)

            else:
                reward = ((1 - self.environment.transaction_cost) * sp - bp) / bp
                print("reward: " + str(reward) + ", state: " + str(date) + ", bp: " + str(bp) + ", sp: " + str(sp))
                # self.model.fit(self.state.value, reward, self.buy_action)

        self.environment.process_epoch_end(date)

    def invoke_buy_order_agent(self):
        # invoking buy order agent with the state of the stock at the same day
        self.__buy_order_agent.start_new_training(self.state.date)

    def start_new_training(self, terminated_by_other_agents=False, evaluation_mode=False):
        print("Buy signal - start new training")
        self.process_next_state()
