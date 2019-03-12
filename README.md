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
## Buy/ Sell signal agents:
This strategy makes use of the price difference between the current price to the previous high/low turning points to determine the buy/ sell signal

## Buy/ Sell order agents:
Different technical indicators, e.g. sma, high/low difference...



Trading system work flow
------------------------
#### 1) Buy Signal Agent
State of stock feeds into BSA, an action is generated as below  

|  Agent Output  |    Action        |
| ------------- |:-------------:|
| BUY      | Trigger Buy Order Agent 2) |
| NOT-BUY      | Go to 1) with next date of state      |

        
#### 2) Buy Order Agent
BOA takes the same state from BSA and determine the buy price of stocks according to the below formula  
   ```
   buy price = ma5 + action / 100 * ma5
   ```
   where action is the output of the agent  
   The order is placed the next day after buy signal is generated in BSA.  
   
|  Buy Price  |    Action        |
| ------------- |:-------------:|
| > next day's low price      | Order is executed, trigger Sell Signal Agent  |
| < next day's low price      | Order is not executed, go back to 1)      |


#### 3) Sell Signal Agent
SSA takes state of stock starting with the day after the buy order is executed, action is generated as below  

|  Agent Output  |    Action        |
| ------------- |:-------------:|
| HOLD      | Go to 3)  |
| SELL      | Trigger Sell Order Agent 2)      |

       
#### 4) Sell Order Agent
SOA takes the same state from SSA and determine the sell price of stocks according to the below formula  
   ```
   sell price = ma5 + action / 100 * ma5  
   ```
   The order is placed the next day after sell signal is generated in SSA.  
   
|  Sell Price  |    Action        |
| ------------- |:-------------:|
| < next day's high price   | Order is executed, position is closed, go to 1)  |
| > next day's high price   | Order is executed at that day's close instead, position is closed, go to 1)     |


       
       
Rewards
-------
#### Buy Signal Agent
rewards = 0 if order is not successfully placed.  
rewards = ((1-transactional cost) * sell price - buy price) / buy price  if position is closed by SOA

#### Buy Order Agent
price diff = buy price - next day's low price  
rewards = exp(-100 * price diff / low price) if price diff >= 0  
rewards = 0 if price diff < 0

#### Sell Signal Agent
rewards = rate of change of close price 

#### Sell Order Agent
price diff = sell price - next day's high price  
rewards = exp(100 * price diff / high price) if price diff <= 0  
rewards = 0 if price diff > 0
