import random

class Agent:
	def __init__(self):
		# parameters/values used by agent to find an optimal decision making sequence
		self.__policy = {}
		self.__gamma = 1
		self.__state_value = {} # maps state -> real number
		self.__action_value = {} # maps (state, action) -> real number

		self.activity_log = [("Start animation", ), ("End animation", )]

	def perceive(self, mdp):
		"""Gives the agent all the information accessible from the environment"""

		self.__perceived_states = mdp.states
		self.__perceived_actions = mdp.actions
		self.__perceived_next_states = mdp.next_states
		self.__perceived_env_rewards = mdp.env_rewards
		self.__perceived_env_probabilities = mdp.env_probabilities

	def perceived_states(self):
		"""Returns all the states available to agent"""

		return self.__perceived_states
		
	def perceived_actions(self, state):
		"""Returns all the actions available to agent at the input state"""

		return self.__perceived_actions[state]
		
	def perceived_next_states(self, state_action_pair):
		"""Returns all the reachable states available to agent from the input state action tuple"""

		return self.__perceived_next_states[state_action_pair]

	def perceived_env_rewards(self):
		"""Returns all the rewards known to agent"""

		return self.__perceived_env_rewards

	def perceived_env_probabilities(self):
		"""Returns all the probabilities known to agent"""

		return self.__perceived_env_probabilities
		
	def is_terminal(self, state):
		"""Returns if the input state is terminal"""

		return False if len(self.__perceived_actions[state]) else True 
		
	def transistion_info(self, state, action, next_state):
		"""Returns the probability and reward of reaching the next state and the value of that state"""

		p = self.__perceived_env_probabilities[(state, action, next_state)]
		r = self.__perceived_env_rewards[(state, action, next_state)]
		v = self.__state_value[next_state]
		return (p, r, v)
		
	def value(self, node):
		"""Returns value of the input state or action node"""

		if isinstance(node, str):
			return self.__state_value[node]
		elif isinstance(node, tuple) and len(node) == 2:
			return self.__action_value[node]
		
	def gamma(self):
		"""Returns the discount factor"""

		return self.__gamma
	
	def policy(self):
		"""Returns the current policy decided by the agent"""

		return self.__policy

	def update_policy(self, state, argmax_action):
		"""Updates the policy of the input state, given the new argmax action"""

		if state in self.__policy:
			prev_best_action = self.__policy[state]
			self.activity_log.insert(-1, ("Policy update", state, prev_best_action, argmax_action))
		self.__policy[state] = argmax_action		

	def update_state_value(self, state, value): 
		"""Sets the value of input state to input value"""

		if state in self.__state_value:
			self.activity_log.insert(-1, ("State Value update", state, self.__state_value[state], value))
		self.__state_value[state] = value
		
	def update_action_value(self, state_action_pair, value):
		"""Sets the value of input action to input value"""

		self.__action_value[state_action_pair] = value
		
	def randomise_policy(self):
		"""Randomizes the agent's policy"""
		
		initial_policy = {}
		log = []
		for state in self.__perceived_states:
			action_choices = self.__perceived_actions[state]
			if action_choices:
				random_best_action = random.choice(action_choices)
				self.update_policy(state, random_best_action)
				log.append((state, random_best_action))
		self.activity_log.insert(-1, ("Policy initiate", tuple(log)))

	def randomise_state_values(self):
		"""Randomizes all state values"""

		initial_value = {}
		log = []
		for state in self.__perceived_states:
			initial_value[state] = random.random()
			self.update_state_value(state, initial_value[state])
			log.append((state, initial_value[state]))
		self.activity_log.insert(-1, ("State Value initiate", tuple(log)))
	
	def randomise_action_values(self):
		"""Randomizes all action values"""

		for state in self.__perceived_states:
			for action in self.__perceived_actions[state]:
				self.update_action_value((state, action), random.random())