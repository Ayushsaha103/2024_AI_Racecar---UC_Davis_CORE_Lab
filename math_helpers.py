
import numpy as np

def vector_magnitude_and_angle(vx, vy):
    # Calculate the magnitude
    magnitude = np.sqrt(vx**2 + vy**2)
    
    # Calculate the directional angle in radians and then convert to degrees
    angle_radians = np.arctan2(vy, vx)
    # angle_degrees = np.degrees(angle_radians)
    
    return magnitude, angle_radians

# def wrap_copy(angle):
#     if angle < -np.pi:
#         w_angle = 2 * np.pi + angle
#     elif angle > np.pi:
#         w_angle = angle - 2 * np.pi
#     else:
#         w_angle = angle

#     return w_angle