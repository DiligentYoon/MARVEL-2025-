"""
A multi-agent worker class for coordinating multi-robots exploration in an indoor environment.

This class manages a group of agents performing collaborative exploration, handling 
their movement, observation, reward calculation, and simulation steps. It supports 
features like collision avoidance, trajectory planning, and performance tracking.

Key functionalities:
- Initializes multiple agents with a shared policy network
- Runs exploration episodes with collision resolution
- Tracks agent locations, headings, and exploration progress
- Generates visualizations of the exploration process
- Calculates rewards and saves episode data

Attributes:
    meta_agent_id (int): Identifier for the meta-agent group
    global_step (int): Current global simulation step
    env (Env): Environment simulation instance
    robot_list (List[Agent]): List of agents in the exploration team
    episode_buffer (List): Buffer for storing episode data
    perf_metrics (dict): Performance metrics for the episode
"""
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.patches import Wedge, FancyArrowPatch

from utils.env import Env
from utils.agent import Agent
from utils.utils import *
from utils.model import PolicyNet
from utils.motion_model import compute_allowable_heading  

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

class MultiAgentWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.fov = FOV
        self.sensor_range = SENSOR_RANGE
        self.dt = DT
        self.sim_steps = NUM_SIM_STEPS

        self.env = Env(global_step, self.fov, self.sensor_range, plot=self.save_image)
        self.n_agents = N_AGENTS

        self.robot_list = [Agent(i, policy_net, self.fov, self.env.angles[i], self.sensor_range, self.device, self.save_image) for i in
                           range(self.n_agents)]

        self.perf_metrics = dict()

    def run_episode(self):
        done = False
        for robot in self.robot_list:
            robot.update_state(self.env.belief_info, self.env.robot_locations[robot.id].copy())

        prev_explored_rate = self.env.explored_rate

        for i in range(MAX_EPISODE_STEP):
            observations = [robot.get_observation() for robot in self.robot_list]
            actions = []
            collisions = []
            for robot_id, robot in enumerate(self.robot_list):
                observation = observations[robot_id]
                velocity, yaw_rate = robot.get_action(observation)
                actions.append((velocity, yaw_rate))
                collisions.append(False)

            # =========== Transition with k(=sim_steps 파라미터) Decimation Step 수행 ===========
            for _ in range(self.sim_steps):
                robot_locations_sim_step = []
                robot_headings_sim_step = []
                for robot_id, robot in enumerate(self.robot_list):
                    velocity, yaw_rate = actions[robot_id]
                    
                    # Update heading
                    new_heading = (robot.heading + yaw_rate * self.dt) % 360

                    # Update location
                    heading_rad = np.radians(new_heading)
                    heading_rad = np.arctan2(np.sin(heading_rad), np.cos(heading_rad))

                    # Check : Location 업데이트, Rotation 업데이트 중 무엇이 선행 ?
                    delta_x = velocity * np.cos(heading_rad) * self.dt
                    delta_y = velocity * np.sin(heading_rad) * self.dt
                    new_location = robot.location + np.array([delta_x, delta_y])

                    # Cartesian 좌표 -> Cell 좌표
                    new_location_cell = get_cell_position_from_coords(new_location, self.env.belief_info)
                    # Collision Detection 로직 : 충돌하지 않은 유효 상태일때만 업데이트
                    if (0 <= new_location_cell[0] < self.env.belief_info.map.shape[1] and
                        0 <= new_location_cell[1] < self.env.belief_info.map.shape[0] and
                        self.env.belief_info.map[new_location_cell[1], new_location_cell[0]] == FREE):
                        # Update environment and agent state
                        self.env.final_sim_step(new_location, robot.id)
                        robot.update_location(new_location)
                        robot.update_heading(new_heading)
                    else:
                        collisions[robot_id] = True
                        new_location = robot.location
                        new_heading = robot.heading
                    
                    # bresenhem 알고리즘 기반, binary occupancy shared map 업데이트: 코드 재사용 가능
                    self.env.update_robot_belief(robot.location, robot.heading)

                    robot_locations_sim_step.append(robot.location)
                    robot_headings_sim_step.append(robot.heading)

                if self.save_image:
                    num_frame = i * self.sim_steps + _
                    self.plot_local_env_sim(num_frame, robot_locations_sim_step, robot_headings_sim_step)


            # Reward calculation
            current_explored_rate = self.env.explored_rate
            reward = (current_explored_rate - prev_explored_rate) * 100 - 0.1 # Exploration reward and time penalty
            prev_explored_rate = current_explored_rate

            if self.env.explored_rate > 0.99:
                done = True
            if done:
                reward += 10

            # At the end of the step, get the next observation
            next_observations = [robot.get_observation() for robot in self.robot_list]

            # Save experience in each agent's episode buffer
            for robot_id, robot in enumerate(self.robot_list):
                obs = observations[robot_id]
                action = actions[robot_id]
                next_obs = next_observations[robot_id]
                robot.save_experience(obs, action, reward, next_obs, done)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)



    # ====================================================================
    # ======================== plot 함수 =================================
    # ====================================================================


    def smooth_heading_change(self, prev_heading, heading, steps=10):
        # Ensure both angles are in the range [0, 360)
        prev_heading = prev_heading % 360
        heading = heading % 360

        # Calculate the angle difference
        diff = heading - prev_heading
        
        # Adjust for the shortest path
        if abs(diff) > 180:
            diff = diff - 360 if diff > 0 else diff + 360

        # Generate intermediate angles
        intermediate_headings = [
            (prev_heading + i * diff / steps) % 360
            for i in range(1, steps)
        ]

        # Ensure the final heading is exactly the target heading
        intermediate_headings.append(heading)
        return intermediate_headings



    def heading_to_vector(self, heading, length=25):
        # Convert heading to vector
        if isinstance(heading, (list, np.ndarray)):
            heading = heading[0]
        heading_rad = np.radians(heading)
        return np.cos(heading_rad) * length, np.sin(heading_rad) * length



    def plot_local_env_sim(self, step, robot_locations, robot_headings):
        plt.switch_backend('agg')
        plt.figure(figsize=(6, 2.5))
        color_list = ['r', 'b', 'g', 'y']
        color_name = ['Red', 'Blue', 'Green', 'Yellow']
        sensing_range = self.sensor_range / CELL_SIZE

        plt.subplot(1, 3, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
          
        for robot in self.robot_list:
            c = color_list[robot.id]
            
            if robot.id == 0:
                nodes = get_cell_position_from_coords(robot.node_coords, robot.map_info)
                plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility, s=8, zorder=2)
                for i, (x, y) in enumerate(nodes):
                    plt.text(x-3, y-3, f'{robot.utility[i]:.0f}', ha='center', va='bottom', fontsize=3, color='blue', zorder=3)
                   
        # Plot frontiers
        global_frontiers = get_frontier_in_map(self.env.belief_info)
        if len(global_frontiers) != 0:
            frontiers_cell = get_cell_position_from_coords(np.array(list(global_frontiers)), self.env.belief_info) #shape is (2,)
            if len(global_frontiers) == 1:
                frontiers_cell = frontiers_cell.reshape(1,2)
            plt.scatter(frontiers_cell[:, 0], frontiers_cell[:, 1], s=1, c='r')       

        plt.subplot(1, 3, 2)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        color_list = ['r', 'b', 'g', 'y']

        for i, (robot, location, heading) in enumerate(zip(self.robot_list, robot_locations, robot_headings)):
            c = color_list[robot.id]
            dx, dy = self.heading_to_vector(heading, length=sensing_range)
            arrow = FancyArrowPatch((location[0], location[1]), (location[0] + dx/1.25, location[1] + dy/1.25),
                                    mutation_scale=10,
                                    color=c,
                                    arrowstyle='-|>')
            plt.gca().add_artist(arrow)
            plt.text(location[0] + 5, location[1] + 5, f'{heading:.0f}°', color=c, fontsize=6, ha='left', va='center', zorder=16)

            robot_location = get_coords_from_cell_position(location, self.env.belief_info)
            trajectory_x = robot.trajectory_x.copy()
            trajectory_y = robot.trajectory_y.copy()
            trajectory_x.append(robot_location[0])
            trajectory_y.append(robot_location[1])
            plt.plot((np.array(trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,
                     (np.array(trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size, c,
                     linewidth=1.2, zorder=1)
            
        # Plot frontiers
        if len(global_frontiers) != 0:
            plt.scatter(frontiers_cell[:, 0], frontiers_cell[:, 1], s=1, c='r')

        # Ground truth data
        plt.subplot(1, 3, 3)
        plt.imshow(self.ground_truth_node_manager.ground_truth_map_info.map, cmap='gray')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        for i, (location, heading) in enumerate(zip(robot_locations, robot_headings)):
            c = color_list[i]
            plt.plot(location[0], location[1], c+'o', markersize=6, zorder=5)
            dx, dy = self.heading_to_vector(heading, length=sensing_range)
            plt.arrow(location[0], location[1], dx, dy, head_width=5, head_length=5, fc=c, ec=c, zorder= 15)

            # Draw cone representing field of vision
            cone = Wedge(center=(location[0], location[1]), r=self.sensor_range / CELL_SIZE, theta1=(heading-self.fov/2), 
                         theta2=(heading+self.fov/2), color=c, alpha=0.5, zorder=10)
            plt.gca().add_artist(cone)
            nodes = get_cell_position_from_coords(self.ground_truth_node_manager.ground_truth_node_coords, self.ground_truth_node_manager.ground_truth_map_info)
            plt.scatter(nodes[:, 0], nodes[:, 1], c=self.ground_truth_node_manager.explored_sign, s=8, zorder=2)

        plt.axis('off')
        robot_headings = [f"{color_name[robot.id]}- {robot.heading:.0f}°" for robot in self.robot_list]
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}\nRobot Headings: {}'.format(
            self.env.explored_rate,
            max([robot.travel_dist for robot in self.robot_list]),
            ', '.join(robot_headings)
        ), fontweight='bold', fontsize=10)
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)
        plt.close()
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)



if __name__ == '__main__':
    from parameter import *
    import torch
    # The input shape for the CNN is (height, width)
    # The map size needs to be an integer for the model input shape.
    map_size = int(UPDATING_MAP_SIZE / CELL_SIZE)
    map_input_shape = (map_size, map_size)
    action_dim = 2 # velocity and yaw_rate

    policy_net = PolicyNet(map_input_shape, EMBEDDING_DIM, action_dim)
    if LOAD_MODEL:
        try:
            checkpoint = torch.load(load_path + '/checkpoint.pth', map_location='cpu')
            policy_net.load_state_dict(checkpoint['policy_model'])
            print('Policy loaded!')
        except FileNotFoundError:
            print("Checkpoint not found, using randomly initialized policy.")

    worker = MultiAgentWorker(0, policy_net, 888, 'cpu', True)
    worker.run_episode()
