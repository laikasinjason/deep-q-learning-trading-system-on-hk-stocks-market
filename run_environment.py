from buy_order_agent import BuyOrderAgent
from buy_signal_agent import BuySignalAgent
from environment import Environment
from progress_recorder import ProgressRecorder
from sell_order_agent import SellOrderAgent
from sell_signal_agent import SellSignalAgent

if __name__ == '__main__':
    print("Starting environment")

    # variables initialization
    progress_recorder = ProgressRecorder()
    env = Environment(None, progress_recorder, 1)
    buy_order_agent = BuyOrderAgent(env)
    buy_signal_agent = BuySignalAgent(env)
    sell_order_agent = SellOrderAgent(env)
    sell_signal_agent = SellSignalAgent(env)

    env.set_agents(buy_signal_agent, sell_signal_agent, buy_order_agent, sell_order_agent)

    env.train_system()
