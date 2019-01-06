import random
import numpy as np

class Agent:
	def __init__(self):
		# parameters/values used by agent to find an optimal decision making sequence
		self.__policy = {}
		self.__gamma = 1
		self.__epsilon = 0.1

		self.__state_value = {} # maps state -> real number
		self.__action_value = {} # maps (state, action) -> real number

		self.activity_log = [("Start animation", ), ("End animation", )]

	def perceive(self, mdp):
		"""Gives the agent all the information accessible from the environment"""
		
		self.__perceived_states = mdp.states
		self.__perceived_actions = mdp.actions
		self.__perceived_start_states = mdp.start_states
		self.__perceived_terminal_states = mdp.terminal_states
		self.sample_next_state = mdp.sample_next_state

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
	
	def policy(self):
		"""Returns the current policy decided by the agent"""
		
		return self.__policy

	def update_policy(self, state, argmax_action):
		"""Updates the policy of the input state, given the new argmax action"""
		
		if (state, argmax_action) in self.__policy:
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

	def generate_sample_episode(self, policy):
		"""Generates a sample episode of agent-environment interaction"""
		
		self.activity_log.insert(-1, ("Begin simulation", ))
		episode = []
		sampled_state = random.choice(self.__perceived_start_states)
		episode.append(sampled_state)
		self.activity_log.insert(-1, ("Sampled state", sampled_state))		

		for i in range(10):
			current_state = sampled_state

			action_choices = self.__perceived_actions[current_state]
			action_probabilities = [policy[(current_state, action)] for action in action_choices]
			sampled_action = np.random.choice(action_choices, 1, None, action_probabilities)[0]
			episode.append(sampled_action)
			self.activity_log.insert(-1, ("Sampled action", (current_state, sampled_action)))

			sampled_state, reward = self.sample_next_state((current_state, sampled_action))
			
			episode.append(reward)
			episode.append(sampled_state)
			self.activity_log.insert(-1, ("Sampled state", sampled_state))
			
			if sampled_state in self.__perceived_terminal_states:
				break

		self.activity_log.insert(-1, ("End simulation", ))	
		return episode