import random
import numpy as np

class Agent:
	def __init__(self):
		# parameters/values used by agent to find an optimal decision making sequence
		self.__policy = {}
		self.__gamma = 1
		self.__epsilon = 0.1
		self.__alpha = 0.1

		self.__state_value = {} # maps state -> real number
		self.__action_value = {} # maps (state, action) -> real number

		self.activity_log = [("Start animation", ), ("End animation", )]

	def perceive(self, mdp):
		"""Gives the agent all the information accessible from the environment"""
		
		self.__perceived_states = mdp.states
		self.__perceived_actions = mdp.actions
		self.__perceived_start_states = mdp.start_states
		self.__perceived_terminal_states = mdp.terminal_states
		self.__sample_next_state = mdp.sample_next_state

	def perceived_states(self):
		"""Returns all the states available to agent"""
		
		return self.__perceived_states
		
	def perceived_actions(self, state):
		"""Returns all the actions available to agent at the input state"""
		
		return self.__perceived_actions[state]
		
	def is_terminal(self, state):
		"""Returns if the input state is terminal"""
		
		return True if state in self.__perceived_terminal_states else False
		
	def value(self, node):
		"""Returns value of the input state or action node"""
		
		if isinstance(node, str):
			return self.__state_value[node]
		elif isinstance(node, tuple) and len(node) == 2:
			return self.__action_value[node]
		
	def gamma(self):
		"""Returns the discount factor"""
		
		return self.__gamma

	def epsilon(self):
		"""Returns the epsilon value"""
		
		return self.__epsilon

	def learning_rate(self):
		"""Returns the learning rate value"""
		
		return self.__alpha
	
	def policy(self):
		"""Returns the current policy decided by the agent"""
		
		return self.__policy

	def update_policy(self, state, argmax_action):
		"""Updates the policy of the input state, given the new argmax action"""

		if (state, argmax_action) in self.__policy:
			# print("hello")
			# prev_best_action = max(self.__policy[(state, argmax_action)], key = self.__policy[(state, argmax_action)].get)[1]
			required_list = {elem[1]: self.__policy[elem] for elem in self.__policy if elem[0] == state}
			prev_best_action = max(required_list, key = required_list.get)
			self.activity_log.insert(-1, ("Policy update", state, prev_best_action, argmax_action))
		action_choices = self.__perceived_actions[state]
		num_choices = len(action_choices)
		if num_choices:
			for action in action_choices:
				self.__policy[(state, action)] = self.__epsilon/num_choices
			self.__policy[(state, argmax_action)] += (1 - self.__epsilon)

	def update_state_value(self, state, value): 
		"""Sets the value of input state to input value"""
		
		if state in self.__state_value:
			self.activity_log.insert(-1, ("State Value update", state, self.__state_value[state], value))
		self.__state_value[state] = value
		
	def update_action_value(self, state_action_pair, value):
		"""Sets the value of input action to input value"""
		
		if state_action_pair in self.__action_value:
			self.activity_log.insert(-1, ("Action Value update", state_action_pair, self.__action_value[state_action_pair], value))
		self.__action_value[state_action_pair] = value

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
		
		initial_value = {}
		log = []
		for state in self.__perceived_states:
			for action in self.__perceived_actions[state]:
				initial_value[(state, action)] = random.random()
				self.update_action_value((state, action), initial_value[(state, action)])
				log.append(((state, action), initial_value[(state, action)]))
		self.activity_log.insert(-1, ("Action Value initiate", tuple(log)))

	def random_initialise_epsilon_soft_policy(self):
		"""Randomizes initial policy of the agent"""
		
		initial_policy = {}
		log = []
		for state in self.__perceived_states:
			action_choices = self.__perceived_actions[state]
			if action_choices:
				random_best_action = random.choice(action_choices)
				self.update_policy(state, random_best_action)
				log.append((state, random_best_action))
		self.activity_log.insert(-1, ("Policy initiate", tuple(log)))

	def choose_action_according_to_current_policy(self, state):
		"""Returns action according to current policy for the input state"""

		sampled_action = None
		if not self.is_terminal(state):
			action_choices = self.__perceived_actions[state]
			action_probabilities = [self.__policy[(state, action)] for action in action_choices]
			sampled_action = np.random.choice(action_choices, 1, None, action_probabilities)[0]
			self.activity_log.insert(-1, ("Sampled action", (state, sampled_action)))
		return sampled_action
	
	def sample_initial_state_and_initial_action(self):
		"""Generates a sample episode of agent-environment interaction"""

		self.activity_log.insert(-1, ("Begin simulation", ))
		sampled_state = random.choice(self.__perceived_start_states)
		self.activity_log.insert(-1, ("Sampled state", sampled_state))
		sampled_action = self.choose_action_according_to_current_policy(sampled_state)
		self.activity_log.insert(-1, ("End simulation", ))
		return sampled_state, sampled_action

	def sample_next_state_and_next_action(self, state_action_pair):
		"""Returns next state, reward to reach it, and the next action according to current policy for the current state and action"""

		self.activity_log.insert(-1, ("Begin simulation", ))
		sampled_state, reward = self.__sample_next_state(state_action_pair)
		self.activity_log.insert(-1, ("Sampled state", sampled_state))
		next_action = self.choose_action_according_to_current_policy(sampled_state)
		self.activity_log.insert(-1, ("End simulation", ))
		return sampled_state, reward, next_action
	
	def argmax_action(self, state):
		"""Returns the action having the maximum Q value"""

		best_action = None
		best_action_value = -float('Inf')
		for action in self.perceived_actions(state):
			exp_val = self.value((state, action))
			if exp_val >= best_action_value:
				best_action = action
				best_action_value = exp_val
		return best_action

	def initialise_policy_according_to_action_values(self):
		"""Initialises policy greedily according to current action values"""

		initial_policy = {}
		log = []
		for state in self.perceived_states():
			if not self.is_terminal(state):
				best_action = self.argmax_action(state)
				self.update_policy(state, best_action)
				log.append((state, best_action))
		self.activity_log.insert(-1, ("Policy initiate", tuple(log)))
