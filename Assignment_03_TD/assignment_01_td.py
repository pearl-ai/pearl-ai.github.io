from agent_knowledge_td import Agent
import transition_graph
from graphics import GraphicsHandler
from environment_mdp import MDP
import numpy as np
			
def sarsa(agent):
	#initialisation
	tolerable_error = 0.01
	agent.randomise_action_values()
	agent.initialise_policy_according_to_action_values() 

	for i in range(5):
		current_state, current_action = agent.sample_initial_state_and_initial_action()
		
		while True:
			next_state, reward, next_action = agent.sample_next_state_and_next_action((current_state, current_action))			

			if agent.is_terminal(next_state):
				break

			curr_action_value = agent.value((current_state, current_action))
			next_action_value = agent.value((next_state, next_action))

			alpha = agent.learning_rate()
			gamma = agent.gamma()
			curr_action_value = curr_action_value + alpha*(reward + gamma*next_action_value - curr_action_value)
			agent.update_action_value((current_state, current_action), curr_action_value)
			agent.update_policy(current_state, agent.argmax_action(current_state))
			
			current_state = next_state
			current_action = next_action

if __name__ == "__main__":
	mdp = MDP()
	mdp.model(transition_graph)
	agent = Agent()
	agent.perceive(mdp)
	sarsa(agent)
	# for a in agent.activity_log:
	# 	print(a)
	GraphicsHandler(agent.activity_log).visualise()