from agent_knowledge_mc import Agent
import transition_graph
from graphics import GraphicsHandler
from environment_mdp import MDP
import numpy as np
	
def argmax_action(state, agent):
	"""Returns the action having the maximum Q value"""
	best_action = None
	best_action_value = -float('Inf')
	for action in agent.perceived_actions(state):
		exp_val = agent.value((state, action))
		if exp_val >= best_action_value:
			best_action = action
			best_action_value = exp_val
	return best_action
			
def on_policy_monte_carlo_control(agent):
	#initialisation
	tolerable_error = 0.01
	agent.randomise_action_values()
	agent.random_initialise_epsilon_soft_policy()
	returns_history = {}
	for i in range(5):
		episode = agent.generate_sample_episode(agent.policy())
		G = 0 #return from end of episode
		episode = episode[0:-1]
		episode = list(zip(episode[0::3], episode[1::3], episode[2::3]))
		episode_returns = {}
		for state, action, reward in reversed(episode):
			G = reward + agent.gamma()*G
			episode_returns[(state, action)] = G
		for state_action_pair, episode_return in episode_returns.items():
			if state_action_pair in returns_history:
				returns_history[state_action_pair].append(episode_return)
			else:
				returns_history[state_action_pair] = [episode_return]
			agent.update_action_value(state_action_pair, np.mean(returns_history[(state_action_pair)]))
		for state in agent.perceived_states():
			if not agent.is_terminal(state):
				agent.update_policy(state, argmax_action(state, agent))

if __name__ == "__main__":
	mdp = MDP()
	mdp.model(transition_graph)
	agent = Agent()
	agent.perceive(mdp)
	on_policy_monte_carlo_control(agent)
	# print (agent.policy())
	GraphicsHandler(agent.activity_log).visualise()