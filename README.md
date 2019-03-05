# deep-q-learning-trading-system-on-hk-stocks-market
Deep q learning on determining buy/sell signal and placing orders



----------
This project is inspired by the paper: 
[A Multi-agent Q-learning Framework for Optimizing Stock Trading Systems](https://link.springer.com/chapter/10.1007/3-540-46146-9_16) 


Trading System Structure
------------------------
The trading system takes 4 agents: buy signal agent, buy order agent, sell order agent, sell signal agent.

  * Buy signal agent makes the long decision by considering the state of stocks in current day
  * Buy order agent determines the buy price after buy signal agent gave the buy signal
  * Sell signal agent makes the short decision ( after holding the stocks, no short sell )
  * Sell order agent determines the sell price after sell signal agent gave the sell signal
  

States of stocks
----------------
### Buy/ Sell signal agents:
This strategy makes use of the price difference between the current price to the previous high/low turning points to determine the buy/ sell signal

### Buy/ Sell order agents:
Different technical indicators, e.g. sma, high/low difference...