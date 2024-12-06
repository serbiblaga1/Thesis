import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import * 

class AltitudeStabilizationController:
    def __init__(self, target_altitude, pid_gains):
        """
        Initialize the altitude stabilization controller.

        :param target_altitude: Desired altitude to reach and maintain.
        :param pid_gains: Gains for the PID controller (a tuple: (Kp, Ki, Kd)).
        """
        self.target_altitude = target_altitude
        self.Kp, self.Ki, self.Kd = pid_gains
        
        # Error terms for PID controller
        self.altitude_error_sum = 0.0
        self.prev_altitude_error = 0.0

    def compute_thrust(self, current_altitude, dt):
        """
        Compute the thrust needed to reach the target altitude.

        :param current_altitude: Current altitude of the drone.
        :param dt: Time delta since the last computation.
        :return: Thrust to apply to stabilize altitude.
        """
        altitude_error = self.target_altitude - current_altitude
        self.altitude_error_sum += altitude_error * dt
        altitude_error_derivative = (altitude_error - self.prev_altitude_error) / dt

        # PID control formula
        thrust = (self.Kp * altitude_error + 
                  self.Ki * self.altitude_error_sum + 
                  self.Kd * altitude_error_derivative)
        
        # Save previous error
        self.prev_altitude_error = altitude_error

        # Normalize thrust output
        thrust = max(0.0, min(1.0, thrust))

        return thrust