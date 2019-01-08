import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, ArrowStyle
from matplotlib import animation
from transition_graph import positions, structure
import numpy as np

# window parameters
matplotlib.rcParams["figure.figsize"]=(10, 5)
matplotlib.rcParams['toolbar'] = 'None'

# axes configuration
fig, ax = plt.subplots()
fig.patch.set_facecolor("#006600")
ax.set_aspect(1)
ax.axis("off")
plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

class GraphicsHandler:
	"""Class to handle all the animation and associated parameters"""

	def __init__(self, activity_log):
		self.activity_log = activity_log
		self.activity_index = 0
		self.activity_start_frame_number = 0
		self.visual_constants = {
			"state_color_light": "#ff3333", 
			"state_color_dark": "#800000", 
			"action_color_light": "#ff1aff", 
			"action_color_dark": "#000080",
			"simulation_color_light": "#ffffff",
			"simulation_color_dark": "#000000",
			"agent_arrow_min_tail_width": 0, 
			"agent_arrow_max_tail_width": 8, 
			"agent_arrow_min_head_width": 4, 
			"agent_arrow_max_head_width": 14, 
			"agent_arrow_min_head_length": 8, 
			"agent_arrow_max_head_length": 14, 
		}
		self.anim_info = {
			"Start animation": (self.start_animation, 50),
			"State Value initiate": (self.initiate_state_value, 30),
			"Action Value initiate": (self.initiate_action_value, 30),
			"State Value update": (self.update_state_value, 30),
			"Action Value update": (self.update_action_value, 30),
			"Policy initiate": (self.initiate_policy, 20),
			"Policy update": (self.update_policy, 20),
			"Begin simulation": (self.begin_simulation, 50),
			"Sampled state": (self.sample_state, 20),
			"Sampled action": (self.sample_action, 20),
			"End simulation": (self.end_simulation, 50),
			"End animation": (self.end_animation, 50)
		}
		self.total_frame_count = sum(map(lambda activity_info: self.anim_info[activity_info[0]][1], activity_log)) - 1

		self.state_nodes = {}
		self.action_nodes = {}
		self.agent_arrows = {}
		self.env_arrows = {}
		self.state_values = {}
		self.action_values = {}
		self.status = ""

		self.x_shift = -1
		self.y_shift = 0
	
	# helper functions
	def rgb2hex(self, rgb):
		"""Converts color from rgb format to hexcode format"""

		r, g, b = rgb
		assert (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255), "Invalid RGB color combination"
		return "#{:02x}{:02x}{:02x}".format(r,g,b)

	def hex2rgb(self, hexcode):
		"""Converts color from hexcode format to rgb format"""

		return np.array(tuple(int(hexcode[i:i+2], 16) for i in (1, 3, 5)))

	def intermediate_color(self, initial_color, blink_color, sub_frame_number, allotted_frames, full_cycle = True):
		"""Returns the intermediate color corresponding to the sub_frame_number/allotted_frames ratio"""

		difference = self.hex2rgb(blink_color) - self.hex2rgb(initial_color)
		initial_color = self.hex2rgb(initial_color)
		completion_ratio = sub_frame_number/(allotted_frames-1)
		if full_cycle:
			color = initial_color + difference*math.sin(math.pi*completion_ratio)**2
		else:
			color = initial_color + difference*math.sin((math.pi/2)*completion_ratio)**2
		return self.rgb2hex(np.array([int(col) for col in color]))

	def intermediate_arrow_size(self, initial_size, final_size, sub_frame_number, allotted_frames):
		"""Returns the intermediate arrow size corresponding to the sub_frame_number/allotted_frames ratio"""

		difference = final_size - initial_size
		completion_ratio = sub_frame_number/(allotted_frames-1)
		return initial_size + difference*completion_ratio

	# animation functions
	def start_animation(self, activity_info, sub_frame_number):
		"""Called at the beginning of the entire animation sequence"""

		self.status.set_text("Starting animation...")

	def end_animation(self, activity_info, sub_frame_number):
		"""Called at the end of the entire animation sequence"""

		self.status.set_text("Animation complete...")

	def initiate_state_value(self, activity_info, sub_frame_number):
		"""Displays the randomly initialised state values"""

		activity_type, activity_tuples = activity_info
		allotted_frames = self.anim_info[activity_type][1]
		color = self.intermediate_color(
					self.visual_constants["state_color_light"],
					self.visual_constants["state_color_dark"],
					sub_frame_number,
					allotted_frames,
				)

		for state_name, initial_value in activity_tuples:
			self.state_nodes[state_name].set_color(color)
			if sub_frame_number == allotted_frames//2:
				self.state_values[state_name].set_text(state_name + "\n" + ("%.2f" % initial_value))
		self.status.set_text("Randomising initial state values")

	def initiate_action_value(self, activity_info, sub_frame_number):
		"""Displays the randomly initialised action values"""

		activity_type, activity_tuples = activity_info
		allotted_frames = self.anim_info[activity_type][1]
		color = self.intermediate_color(
					self.visual_constants["action_color_light"],
					self.visual_constants["action_color_dark"],
					sub_frame_number,
					allotted_frames,
				)

		for (state_name, action_name), initial_value in activity_tuples:
			self.action_nodes[(state_name, action_name)].set_color(color)
			if sub_frame_number == allotted_frames//2:
				self.action_values[(state_name, action_name)].set_text(action_name + "\n" + ("%.2f" % initial_value))
		self.status.set_text("Randomising initial action values")
			
	def initiate_policy(self, activity_info, sub_frame_number):
		"""Displays the randomly initialised policy (by altering arrow thickness)"""

		activity_type, activity_tuples = activity_info
		allotted_frames = self.anim_info[activity_type][1]
		
		increasing_tail_width = self.intermediate_arrow_size(
									self.visual_constants["agent_arrow_min_tail_width"],
									self.visual_constants["agent_arrow_max_tail_width"],
									sub_frame_number,
									allotted_frames,
								)
		increasing_head_width = self.intermediate_arrow_size(
									self.visual_constants["agent_arrow_min_head_width"],
									self.visual_constants["agent_arrow_max_head_width"],
									sub_frame_number,
									allotted_frames,
								)
		increasing_head_length = self.intermediate_arrow_size(
									self.visual_constants["agent_arrow_min_head_length"],
									self.visual_constants["agent_arrow_max_head_length"],
									sub_frame_number,
									allotted_frames,
								)
		
		for state_name, action_name in activity_tuples:
			self.agent_arrows[(state_name, action_name)].set_arrowstyle("simple", tail_width=increasing_tail_width, head_width=increasing_head_width, head_length=increasing_head_length)
		self.status.set_text("Randomising initial policy")

	def update_state_value(self, activity_info, sub_frame_number):
		"""Used to display state value updates"""

		activity_type, state_name, old_value, new_value = activity_info
		allotted_frames = self.anim_info[activity_type][1]
		color = self.intermediate_color(
					self.visual_constants["state_color_light"],
					self.visual_constants["state_color_dark"],
					sub_frame_number,
					allotted_frames,
				)
		self.state_nodes[state_name].set_color(color)
		if sub_frame_number == allotted_frames//2:
			self.state_values[state_name].set_text(state_name + "\n" + ("%.2f" % new_value))
		self.status.set_text("Updating state value")

	def update_action_value(self, activity_info, sub_frame_number):
		"""Used to display action value updates"""

		activity_type, state_action_tuple, old_value, new_value = activity_info
		allotted_frames = self.anim_info[activity_type][1]
		color = self.intermediate_color(
					self.visual_constants["action_color_light"],
					self.visual_constants["action_color_dark"],
					sub_frame_number,
					allotted_frames,
				)

		self.action_nodes[state_action_tuple].set_color(color)
		if sub_frame_number == allotted_frames//2:
			self.action_values[state_action_tuple].set_text(state_action_tuple[1] + "\n" + ("%.2f" % new_value))
		self.status.set_text("Updating action value")
		
	def update_policy(self, activity_info, sub_frame_number):
		"""Used to display policy updates (by altering arrow thickness)"""

		activity_type, state_name, old_action_name, new_action_name = activity_info
		allotted_frames = self.anim_info[activity_type][1]
		ratio = sub_frame_number/allotted_frames
		increasing_tail_width = self.intermediate_arrow_size(
									self.visual_constants["agent_arrow_min_tail_width"],
									self.visual_constants["agent_arrow_max_tail_width"],
									sub_frame_number,
									allotted_frames,
								)
		increasing_head_width = self.intermediate_arrow_size(
									self.visual_constants["agent_arrow_min_head_width"],
									self.visual_constants["agent_arrow_max_head_width"],
									sub_frame_number,
									allotted_frames,
								)
		increasing_head_length = self.intermediate_arrow_size(
									self.visual_constants["agent_arrow_min_head_length"],
									self.visual_constants["agent_arrow_max_head_length"],
									sub_frame_number,
									allotted_frames,
								)
		decreasing_tail_width = self.intermediate_arrow_size(
									self.visual_constants["agent_arrow_max_tail_width"],
									self.visual_constants["agent_arrow_min_tail_width"],
									sub_frame_number,
									allotted_frames,
								)
		decreasing_head_width = self.intermediate_arrow_size(
									self.visual_constants["agent_arrow_max_head_width"],
									self.visual_constants["agent_arrow_min_head_width"],
									sub_frame_number,
									allotted_frames,
								)
		decreasing_head_length = self.intermediate_arrow_size(
									self.visual_constants["agent_arrow_max_head_length"],
									self.visual_constants["agent_arrow_min_head_length"],
									sub_frame_number,
									allotted_frames,
								)
		if old_action_name != new_action_name:
			self.agent_arrows[(state_name, new_action_name)].set_arrowstyle("simple", tail_width=increasing_tail_width, head_width=increasing_head_width, head_length=increasing_head_length)
			self.agent_arrows[(state_name, old_action_name)].set_arrowstyle("simple", tail_width=decreasing_tail_width, head_width=decreasing_head_width, head_length=decreasing_head_length)
		self.status.set_text("Updating policy")

	def begin_simulation(self, activity_info, sub_frame_number):
		"""Alters all node colors before beginning the algorithm simulation"""

		activity_type, = activity_info
		allotted_frames = self.anim_info[activity_type][1]
		for state_name, state_node in self.state_nodes.items():
			start_color = self.rgb2hex(tuple([int(255*state_node.get_facecolor()[i]) for i in range(3)]))
			state_color = self.intermediate_color(
					start_color,
					self.visual_constants["simulation_color_light"],
					sub_frame_number,
					allotted_frames,
					full_cycle = False,
				)
			state_node.set_color(state_color)
		for state_action_tuple, action_node in self.action_nodes.items():
			start_color = self.rgb2hex(tuple([int(255*action_node.get_facecolor()[i]) for i in range(3)]))
			action_color = self.intermediate_color(
						start_color,
						self.visual_constants["simulation_color_light"],
						sub_frame_number,
						allotted_frames,
						full_cycle = False,
					)
			action_node.set_color(action_color)

		self.status.set_text("Beginning simulation")

	def sample_state(self, activity_info, sub_frame_number):
		"""Changes the state color during simulation"""
		activity_type, state_name = activity_info
		allotted_frames = self.anim_info[activity_type][1]
		color = self.intermediate_color(
					self.visual_constants["simulation_color_light"],
					self.visual_constants["simulation_color_dark"],
					sub_frame_number,
					allotted_frames,
					full_cycle = False,
				)

		self.state_nodes[state_name].set_color(color)
		self.status.set_text("Sampling next state")

	def sample_action(self, activity_info, sub_frame_number):
		"""Changes the action color during simulation"""

		activity_type, state_action_tuple = activity_info
		allotted_frames = self.anim_info[activity_type][1]
		color = self.intermediate_color(
					self.visual_constants["simulation_color_light"],
					self.visual_constants["simulation_color_dark"],
					sub_frame_number,
					allotted_frames,
					full_cycle = False,
				)
		self.action_nodes[state_action_tuple].set_color(color)
		self.status.set_text("Sampling next action")

	def end_simulation(self, activity_info, sub_frame_number):
		"""Sets all nodes back to initial colors at the end of simulation"""

		activity_type, = activity_info
		allotted_frames = self.anim_info[activity_type][1]
		for state_name, state_node in self.state_nodes.items():
			start_color = self.rgb2hex(tuple([int(255*state_node.get_facecolor()[i]) for i in range(3)]))
			state_color = self.intermediate_color(
						start_color,
						self.visual_constants["state_color_light"],
						sub_frame_number,
						allotted_frames,
						full_cycle = False,
					)
			state_node.set_color(state_color)

		for state_action_tuple, action_node in self.action_nodes.items():
			start_color = self.rgb2hex(tuple([int(255*action_node.get_facecolor()[i]) for i in range(3)]))
			action_color = self.intermediate_color(
						start_color,
						self.visual_constants["action_color_light"],
						sub_frame_number,
						allotted_frames,
						full_cycle = False,
					)
			action_node.set_color(action_color)

		self.status.set_text("Ending simulation")

	def visualise(self):
		"""Orchestrates the entire animation sequence"""

		# setting up the artists
		def initialise():
			"""Sets up all the nodes and arrows on the window"""

			for state_name, center in positions["state_nodes"].items():
				center = tuple(np.array(center) + (self.x_shift, self.y_shift))
				state_node = Circle(center, 1.5, color = self.visual_constants["state_color_light"], zorder = 1)
				self.state_nodes[state_name] = state_node
				ax.add_artist(state_node)
			for state_action_tuple, center in positions["action_nodes"].items():
				center = tuple(np.array(center) + (self.x_shift, self.y_shift))
				action_node = Circle(center, 0.8, color = self.visual_constants["action_color_light"])
				self.action_nodes[state_action_tuple] = action_node
				ax.add_artist(action_node)
			for state_name in structure:
				for action_name in structure[state_name]:
					arrow_start = self.state_nodes[state_name].center
					arrow_end = self.action_nodes[(state_name, action_name)].center
					arrow_style = "simple, tail_width=0.1, head_width=4, head_length=8"
					arrow = FancyArrowPatch(arrow_start, arrow_end, connectionstyle = "arc3, rad=-.4", arrowstyle = arrow_style, color = "k", shrinkA = 30, shrinkB = 16)
					arrow.set_color("#66ff66")
					self.agent_arrows[(state_name, action_name)] = arrow
					ax.add_artist(arrow)
			for state_name in structure:
				for action_name in structure[state_name]:
					for next_state_name in structure[state_name][action_name]:
						arrow_start = self.action_nodes[(state_name, action_name)].center
						arrow_end = self.state_nodes[next_state_name].center
						arrow_style = "simple, tail_width=0.1, head_width=4, head_length=8"
						arrow = FancyArrowPatch(arrow_start, arrow_end, linestyle=(0, (2,5)), connectionstyle = "arc3, rad=-.4", arrowstyle = arrow_style, color = "k", shrinkA = 16, shrinkB = 30)
						arrow.set_color("#66ff66")
						self.env_arrows[(state_name, action_name, next_state_name)] = arrow
						ax.add_artist(arrow)
			for state_name, node in self.state_nodes.items():
				self.state_values[state_name] = ax.text(node.center[0], node.center[1], state_name, horizontalalignment='center', verticalalignment='center', zorder = 5, color = "w")

			for state_action_tuple, node in self.action_nodes.items():
				self.action_values[state_action_tuple] = ax.text(node.center[0], node.center[1], state_action_tuple[1], horizontalalignment='center', verticalalignment='center', zorder = 5, color = "w")

			self.status = ax.text(7, 0, "start", horizontalalignment='left', verticalalignment='center', zorder = 5, color = "k", fontsize = "large", fontweight = "bold")
			
			return tuple(
				list(self.state_nodes.values()) +
				list(self.state_values.values()) +
				list(self.agent_arrows.values()) +
				list(self.env_arrows.values()) +
				[self.status])

		# main animation control
		def animation_seq(frame_number):
			"""Invokes all the other animation functions"""

			ax.set_xlim(-20, 20)
			ax.set_ylim(-10, 10)

			activity_type = self.activity_log[self.activity_index][0]
			activity_end_frame_number = self.activity_start_frame_number + self.anim_info[activity_type][1]
			
			activity_info = self.activity_log[self.activity_index] 
			sub_frame_number = frame_number - self.activity_start_frame_number
			animate_func = self.anim_info[activity_type][0]

			animate_func(activity_info, sub_frame_number)
			
			if frame_number == activity_end_frame_number - 1:
				self.activity_index += 1
				self.activity_start_frame_number = activity_end_frame_number
			
			return tuple(
				list(self.state_nodes.values()) +
				list(self.action_nodes.values()) +
				list(self.state_values.values()) +
				list(self.action_values.values()) +
				list(self.agent_arrows.values()) +
				list(self.env_arrows.values()) +
				[self.status]
			)

		anim = animation.FuncAnimation(fig, animation_seq, init_func = initialise, frames=self.total_frame_count, interval=10, blit=True, repeat = False)
		plt.show()
