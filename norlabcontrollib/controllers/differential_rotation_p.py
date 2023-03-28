from norlabcontrollib.controllers.controller import Controller
import numpy as np
import math


class DifferentialRotationP(Controller):
    def __init__(self, parameter_map):
        super().__init__(parameter_map)
        self.p_gain = parameter_map["p_gain"]
        self.angular_distance_to_goal = 100000

    def compute_angular_error(self):
        angular_error = self.goal_angle - self.current_angle
        if angular_error > math.pi:
            angular_error -= 2 * math.pi
        elif angular_error < -math.pi:
            angular_error += 2 * math.pi
        return angular_error

    def compute_command_vector(self, state):
        self.current_angle = state[5]
        self.goal_angle = self.path.poses[0, 5]
        error = self.compute_angular_error()
        self.angular_distance_to_goal = abs(error)
        return np.array([0, min(self.p_gain * error, self.maximum_angular_velocity)])

    def update_path(self, new_path):
        super().update_path(new_path)
        self.angular_distance_to_goal = 100000
