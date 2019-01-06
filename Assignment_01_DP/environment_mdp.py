import random
import numpy as np

class MDP:
	def __init__(self):
		# Initialise MDP as a '4-tuple' ==> (S, A, P, R)
		self.states = []
		self.actions = {}
		self.env_probabilities = {}
		self.env_rewards = {}

	def model(self, tg):
		# function to fill the (S, A, P, R) properties of the MDP model using a transition graph
		self.start_states = tg.start_states
		self.terminal_states = [state for state in tg.structure.keys() if len(tg.structure[state]) == 0]
		self.next_states = {}
		self.states = list(tg.structure.keys())
		self.actions = {state: list(tg.structure[state].keys()) for state in self.states}
		for state, state_info in tg.structure.items():
			for action, action_info in state_info.items():
				self.next_states[(state, action)] = list(action_info.keys())
				for next_state, next_state_info in action_info.items():
					self.env_probabilities[(state, action, next_state)] = next_state_info[0]
					self.env_rewards[(state, action, next_state)] = next_state_info[1]

	def sample_next_state(self, state_action_pair):
		state_choices = self.next_states[state_action_pair]
		state_probabilities = [self.env_probabilities[(state_action_pair[0], state_action_pair[1], state)] for state in state_choices]
		sampled_next_state = np.random.choice(state_choices, 1, None, state_probabilities)[0]
		reward = self.env_rewards[(state_action_pair[0], state_action_pair[1], sampled_next_state)]
		return (sampled_next_state, reward)