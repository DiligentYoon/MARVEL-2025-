"""
Agent class for multi-robot exploration using a policy network.

This class manages an individual agent's state, movement and mapping in a multi-robot exploration environment. It handles key functionalities 
such as:
- Tracking agent location, heading, and travel distance
- Updating map and frontier information
- Managing graph-based exploration
- Generating observations for policy network
- Selecting next waypoints
- Saving episode data for learning

Attributes:
    id (int): Unique identifier for the agent
    policy_net (torch.nn.Module): Neural network for action selection
    location (np.ndarray): Current agent location
    heading (float): Current agent heading in degrees
    sensor_range (float): Maximum sensing range of the agent
"""
import copy
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from skimage.draw import polygon as sk_polygon

from utils.utils import *
from parameter import *

class Agent:
    def __init__(self, id, policy_net, fov, heading, sensor_range, device='cpu', plot=False):
        self.id = id
        self.device = device
        self.plot = plot
        self.policy_net = policy_net
        self.fov = fov
        self.sensor_range = sensor_range

        # location and global map
        self.location = None
        self.map_info = None

        # Motion parameters
        self.max_velocity = VELOCITY
        self.max_yawrate = YAW_RATE

        # map related parameters
        self.cell_size = CELL_SIZE
        self.updating_map_size = UPDATING_MAP_SIZE

        # map and updating map
        self.map_info = None
        self.updating_map_info = None

        # metrics
        self.travel_dist = 0

        # Heading info
        angle = 0 if heading == 360 else heading
        self.heading = angle

        self.episode_buffer = []

        if self.plot:
            self.trajectory_x = []
            self.trajectory_y = []


    def update_map(self, map_info):
        # no need in training because of shallow copy
        self.map_info = map_info

    def update_updating_map(self, location):
        self.updating_map_info = self.get_updating_map(location)

    def update_location(self, location):
        if self.location is None:
            dist = 0
        else:
            dist = np.linalg.norm(self.location - location)
        self.travel_dist += dist

        self.location = location
            
        if self.plot:
            self.trajectory_x.append(location[0])
            self.trajectory_y.append(location[1])

    def update_heading(self, heading):
        # Update heading data
        self.heading = heading
        
    def get_updating_map(self, location):
        updating_map_origin_x = (location[0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[1] - self.updating_map_size / 2)

        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        min_x = self.map_info.map_origin_x
        min_y = self.map_info.map_origin_y
        max_x = (self.map_info.map_origin_x + self.cell_size * (self.map_info.map.shape[1] - 1))
        max_y = (self.map_info.map_origin_y + self.cell_size * (self.map_info.map.shape[0] - 1))

        if updating_map_origin_x < min_x:
            updating_map_origin_x = min_x
        if updating_map_origin_y < min_y:
            updating_map_origin_y = min_y
        if updating_map_top_x > max_x:
            updating_map_top_x = max_x
        if updating_map_top_y > max_y:
            updating_map_top_y = max_y

        updating_map_origin_x = (updating_map_origin_x // self.cell_size + 1) * self.cell_size
        updating_map_origin_y = (updating_map_origin_y // self.cell_size + 1) * self.cell_size
        updating_map_top_x = (updating_map_top_x // self.cell_size) * self.cell_size
        updating_map_top_y = (updating_map_top_y // self.cell_size) * self.cell_size

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, self.map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, self.map_info)

        updating_map = self.map_info.map[
                    updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1]+1,
                    updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0]+1]

        updating_map_info = MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

        return updating_map_info

    def update_state(self, map_info, location):
        self.update_map(map_info)
        self.update_location(location)
        self.update_updating_map(self.location)


    def get_observation(self):
        # Return the agent's local occupancy grid map.
        # The map is centered around the agent's location.
        shared_map = self.map_info.map
        local_map = self.updating_map_info.map
        # The input to the policy network should be a tensor.
        shared_map_tensor = torch.FloatTensor(shared_map).unsqueeze(0).unsqueeze(0).to(self.device)
        local_map_tensor = torch.FloatTensor(local_map).unsqueeze(0).unsqueeze(0).to(self.device)
        return shared_map_tensor, local_map_tensor


    def get_action(self, observation):
        if observation is None:
            # Return a no-op action if observation is not ready
            return 0.0, 0.0 # velocity, yaw_rate

        with torch.no_grad():
            action_tensor = self.policy_net(observation)

        action = action_tensor.squeeze().cpu().numpy()

        # The action is assumed to be [velocity_scale, yaw_rate_scale]
        # Convert action values from [-1, 1] to physical values
        velocity_scale = action[0]
        yaw_rate_scale = action[1]

        # Scale the actions by the maximum values
        final_velocity = velocity_scale * self.max_velocity
        final_yaw_rate = yaw_rate_scale * self.max_yawrate

        return final_velocity, final_yaw_rate
    
    
    
    def heading_to_vector(self, heading, length=25):
        # Convert heading to vector
        if isinstance(heading, (list, np.ndarray)):
            heading = heading[0]
        heading_rad = np.radians(heading)
        return np.cos(heading_rad) * length, np.sin(heading_rad) * length
    
    def create_sensing_mask(self, location, heading, mask):

        location_cell = get_cell_position_from_coords(location, self.map_info)
        # Create a Point for the robot's location
        robot_point = Point(location_cell)

        # Calculate the angles for the sector
        start_angle = (heading - self.fov / 2 + 360) % 360
        end_angle = (heading + self.fov / 2) % 360

        # Create points for the sector
        sector_points = [robot_point]
        if start_angle <= end_angle:
            angle_range = np.linspace(start_angle, end_angle, 20)
        else:
            angle_range = np.concatenate([np.linspace(start_angle, 360, 10), np.linspace(0, end_angle, 10)])
        for angle in angle_range:
            x = location_cell[0] + self.sensor_range/CELL_SIZE * np.cos(np.radians(angle))
            y = location_cell[1] + self.sensor_range/CELL_SIZE * np.sin(np.radians(angle))
            sector_points.append(Point(x, y))
        sector_points.append(robot_point) 
        # Create the sector polygon
        sector = Polygon(sector_points)

        x_coords, y_coords = sector.exterior.xy
        y_coords = np.rint(y_coords).astype(int)
        x_coords = np.rint(x_coords).astype(int)
        rr, cc = sk_polygon(
                [int(round(y)) for y in y_coords],
                [int(round(x)) for x in x_coords],
                shape=mask.shape
            )
        
        free_connected_map = get_free_and_connected_map(location, self.map_info)

        mask[rr, cc] = (free_connected_map[rr, cc] == free_connected_map[location_cell[1], location_cell[0]])
       
        return mask
    
    def calculate_overlap_reward(self, current_robot_location, all_robots_locations, robot_headings_list):
        ## Robot heading list in degrees
        current_sensing_mask = np.zeros_like(self.map_info.map)
        other_robot_sensing_mask = np.zeros_like(self.map_info.map)
        
        for robot_location, robot_heading in zip(all_robots_locations, robot_headings_list):
            if np.array_equal(current_robot_location, robot_location):       
                current_sensing_mask = self.create_sensing_mask(robot_location, robot_heading, current_sensing_mask) 
            else:
                other_robot_sensing_mask = self.create_sensing_mask(robot_location, robot_heading, other_robot_sensing_mask)

        # Keep cell value of 1 only for cells that hold a value of 255 in self.global_map_info.map
        current_free_area_size = np.sum(current_sensing_mask)
        unique_sensing_mask = np.logical_and(current_sensing_mask == 1, other_robot_sensing_mask == 0).astype(int)
        # Compute the number of cells that have a value of 1 in current_sensing_mask and 0 in other_robot_sensing_mask
        current_free_area_not_scanned_size = np.sum(unique_sensing_mask)

        overlap_reward = np.square(current_free_area_not_scanned_size / current_free_area_size)     

        
        return overlap_reward
        
    def save_experience(self, obs, action, reward, next_obs, done):
        self.episode_buffer.append((obs, action, reward, next_obs, done))