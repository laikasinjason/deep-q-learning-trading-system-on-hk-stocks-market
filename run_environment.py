from buy_order_agent import BuyOrderAgent
from buy_signal_agent import BuySignalAgent
from environment import Environment
from sell_order_agent import SellOrderAgent
from sell_signal_agent import SellSignalAgent

if __name__ == '__main__':
    print("Starting environment")

    # variables initialization
    env = Environment()
    buy_order_agent = BuyOrderAgent(env)
    buy_signal_agent = BuySignalAgent(env, 1)
    sell_order_agent = SellOrderAgent(env)
    sell_signal_agent = SellSignalAgent(env)
    buy_signal_agent.set_buy_order_agent(buy_order_agent)
    buy_order_agent.set_buy_signal_agent(buy_signal_agent)
    buy_order_agent.set_sell_signal_agent(sell_signal_agent)
    sell_signal_agent.set_sell_order_agent(sell_order_agent)
    sell_order_agent.set_buy_signal_agent(buy_signal_agent)

    buy_signal_agent.start_new_training()
