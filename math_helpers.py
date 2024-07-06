
import numpy as np
import math

def vector_magnitude_and_angle(vx, vy):
    # Calculate the magnitude
    magnitude = np.sqrt(vx**2 + vy**2)
    
    # Calculate the directional angle in radians and then convert to degrees
    angle_radians = np.arctan2(vy, vx)
    # angle_degrees = np.degrees(angle_radians)
    
    return magnitude, angle_radians

def steering_reward(delta, d_delta_dt, previous_delta, time_step):
    # Parameters for reward calculation
    high_reward = 10.0
    low_reward = -10.0
    tolerance = 0.1  # Tolerance for acceptable steering angle change
    
    # Calculate the change in steering angle
    change_in_delta = abs(delta - previous_delta)
    
    # Reward logic
    if change_in_delta <= tolerance:
        reward = high_reward  # High reward for minimal change
    else:
        reward = low_reward * change_in_delta  # Penalize large changes
    
    return reward

# # Example usage
# delta = 0.5
# d_delta_dt = 0.05
# previous_delta = 0.48
# time_step = 0.1

# reward = steering_reward(delta, d_delta_dt, previous_delta, time_step)
# print(f"Reward: {reward}")


def steering_reward_continuous_normalized(delta, d_delta_dt, previous_delta, time_step):
    # Parameters for reward calculation
    max_delta_change = math.radians(30) * 2  # Max possible change in steering angle (in radians)
    penalty_factor = 50.0  # Factor to control the steepness of the penalty
    
    # Calculate the change in steering angle
    change_in_delta = abs(delta - previous_delta)
    
    # Reward logic using a smooth penalty function
    penalty = penalty_factor * (change_in_delta ** 2)  # Quadratic penalty
    
    # Normalize the penalty to be between 0 and 1
    normalized_penalty = penalty / (penalty_factor * (max_delta_change ** 2))
    
    # Calculate the normalized reward
    normalized_reward = 1.0 - normalized_penalty
    
    # Ensure the reward is within the range [0, 1]
    normalized_reward = max(0.0, min(1.0, normalized_reward))
    
    return normalized_reward