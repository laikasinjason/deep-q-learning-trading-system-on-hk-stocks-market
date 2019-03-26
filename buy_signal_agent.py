from agent import Agent
from model import SignalModel


class BuySignalAgent(Agent):
    def __init__(self, environment):
        super().__init__(environment)
        # high turning point 5*8, low turning point 5*8, technical indicator 4*8
        self.model = SignalModel(2, 112)
        self.state = None  # save the state to be trained
        self.buy_action = None  # save the action needed to pass to fit method

    def process_next_state(self, date):
        print("Buy signal - processing date: " + str(date))
        self.state, self.buy_action = self.produce_state_and_get_action(date)

        if self.state is None or self.buy_action is None:
            self.environment.process_epoch_end(None, True)
        elif self.buy_action:
            # invoking buy order agent with the state of the stock at the same day
            self.environment.invoke_buy_order_agent()

    def update_reward(self, from_buy_order_agent, date, bp=None, sp=None):
        if not self.environment.get_evaluation_mode():
            if from_buy_order_agent:
                reward = 0
                print("reward: " + str(reward) + ", state: " + str(self.state.date) + ", bp: " + str(bp) + ", sp: " + str(sp))
                self.model.fit(self.state.value, reward, self.buy_action)

            else:
                reward = ((1 - self.environment.transaction_cost) * sp - bp) / bp
                print("reward: " + str(reward) + ", state: " + str(self.state.date) + ", bp: " + str(bp) + ", sp: " + str(sp))
                self.model.fit(self.state.value, reward, self.buy_action)

        self.environment.process_epoch_end(date)
