from sell_order_agent import SellOrderAgent
from buy_order_agent import BuyOrderAgent
from sell_signal_agent import SellSignalAgent
from buy_signal_agent import BuySignalAgent
from environment import Environment



if __name__ == '__main__':
    print("Starting environment")
    
    # variables initialization
    env = Environment()
    sellOrderAgent = SellOrderAgent(env)
    buyOrderAgent = BuyOrderAgent(env)
    sellSignalAgent = SellSignalAgent(env)
    buySignalAgent = BuySignalAgent(env, 1)
    buySignalAgent.set_buy_order_agent(buyOrderAgent)
    sellSignalAgent.set_sell_order_agent(sellOrderAgent)
    buyOrderAgent.set_buy_signal_agent(buySignalAgent)
    buyOrderAgent.set_sell_signal_agent(sellSignalAgent)
    sellOrderAgent.set_buy_signal_agent(buySignalAgent)
    
    buySignalAgent.start_new_training()