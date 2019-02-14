import random


class Environment:
	def __init__(self):
		self.data = None
		self.states = None
		
	@staticmethod
	def get_random_action(agent):
		if isinstance(agent, BuyOrderAgent):
			action = random.randint(0,7)
		elif isinstance(agent, SellOrderAgent):
			action = random.randint(0,7)
		elif isinstance(agent, BuySignalAgent):
			action = random.randint(0,2)
		elif isinstance(agent, SellSignalAgent):
			action = random.randint(0,2)
		
	def generate_sell_signal_states(self):
		s = "sellSignalStates"
		return s
		
	def generate_buy_signal_states(self, isFirst):
		if isFirst:
			# randomly pick a day from dataset
			s = "buySignalStates - first"
		else:
			s = "buySignalStates"
		return s
		
	def generate_sell_order_states(self):
		s = "sellOrderStates"
		return s
		
	def generate_buy_order_states(self):
		s = "buyOrderStates"
		return s
		
	def load_data(self):
		# load from csv/ db
		pass
		
	def get_market_data_by_date_of_state(self, state):
		pass
		
	def produce_state(agent, isFirst):
		if isinstance(agent, BuyOrderAgent):
			s = self.generate_buy_order_states()
		elif isinstance(agent, SellOrderAgent):
			s = self.generate_sell_order_states()
		elif isinstance(agent, BuySignalAgent):
			s = self.generate_buy_signal_states(isFirst)
		elif isinstance(agent, SellSignalAgent):
			s = self.generate_sell_signal_states()
		
		print(s)
		return s
transaction_cost = 1
