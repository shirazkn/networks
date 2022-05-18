import types
import cvxpy as cvx
import dccp
from copy import deepcopy
import numpy as np
import numpy.linalg as la
from scipy.linalg import block_diag
from functions import misc, plot, mitigation_strategies, sensor, config


def plot_red(self, **kwargs):
    if self.gps_timer.get_elapsed_time() < 0.1:
        plot.plot_point(misc.tuple_from_col_vec(
            self._pos + misc.column([0.1, 0.1])
        ), color=(0.95, 0.1, 0.2), s=30, edgecolor=(0.85, 0.05, 0.15))


def get_subvector(vec, index):
    # Gets index-th column subvector of length 2 from a column vector
    return vec[2*index:2*index+2]


class Attacker:
    def __init__(self):
        self.compromised_drones = []
        self.attack_vector = []
        self.optimization_window = 4
        self.prediction_timestep = config.GPS_TIMEOUT*4

        self.gps_residual_limit = 100
        self.ins_residual_limit = 100
        self.range_residual_limit = 50

    def add_compromised_drone(self, drone):
        self.compromised_drones.append(drone.name)

        # Whenever drone calls get_gps, call get_fake_gps instead:
        drone.get_gps_measurement = types.MethodType(self.get_fake_gps, drone)

        # For plotting purposes:
        drone.plot = types.MethodType(plot_red, drone)

    def get_fake_gps(self, obj: mitigation_strategies.DroneWithRangeMitigation):
        # Must call design_attack first!
        index = self.compromised_drones.index(obj.name)
        return obj._pos + misc.white_noise(obj.gps_cov) + get_subvector(self.attack_vector, index)

    def design_attack(self, drones: [mitigation_strategies.DroneWithRangeMitigation]):
        # For now this method has access to entire graph (for convenience)
        design_vector = cvx.Variable((len(self.compromised_drones)*2*self.optimization_window, 1))
        position_errors = {}
        velocity_errors = {}
        for name in drones:
            position_errors[name] = deepcopy(drones[name].ekf.x[0:2])
            velocity_errors[name] = deepcopy(drones[name].ekf.x[2:4])

        # EKF/Prediction matrices are kept common for all drones, for simplicity
        temp_drone = drones[self.compromised_drones[0]]
        I_2 = np.identity(2)
        # C_matrix = np.identity(4)
        P = temp_drone.ekf.P
        measurement_noise = block_diag(temp_drone.gps_cov, temp_drone.ins_cov)
        gps_cov_scalar = 2 * temp_drone.gps_cov[0][0]
        ins_cov_scalar = 2 * temp_drone.ins_cov[0][0]

        K = P @ np.linalg.pinv((measurement_noise + P))

        M_1 = I_2*self.prediction_timestep - K[0:2, 2:4]
        M_2 = I_2*self.prediction_timestep - K[2:4, 2:4]
        K_P = K[0:2, 0:2]
        K_VP = K[0:2, 2:4]
        M_3 = np.identity(4) - K

        constraints = []
        for _ in range(1, self.optimization_window+1):
            for name in drones:
                if name in self.compromised_drones:
                    del_a = get_subvector(design_vector, self.compromised_drones.index(name))
                    position_errors[name] = position_errors[name] + M_1 @ velocity_errors[name] + K_P @ del_a
                    velocity_errors[name] = M_2 @ velocity_errors[name] + K_VP @ del_a

                    position_res = position_errors[name] + del_a
                    velocity_res = velocity_errors[name]
                    constraints.append((cvx.norm(position_res / gps_cov_scalar)) <= self.gps_residual_limit)
                    constraints.append((cvx.norm(velocity_res / ins_cov_scalar)) <= self.ins_residual_limit)

                    for neighbor in drones[name].neighbors:
                        del_e = (position_errors[neighbor.name] - position_errors[name])
                        constraints.append((del_e.T @ (neighbor._pos - drones[name]._pos)*2)
                                           <= self.range_residual_limit)

                else:
                    error = M_3 @ np.concatenate([position_errors[name], velocity_errors[name]])
                    position_errors[name] = error[0:2]
                    velocity_errors[name] = error[2:4]

        objective_errors = []
        for name in self.compromised_drones:
            objective_errors.append(cvx.norm(position_errors[name]))

        optimization_problem = cvx.Problem(cvx.Maximize(cvx.max(cvx.hstack(objective_errors))), constraints)
        optimization_problem.solve(method='dccp')

        self.attack_vector = np.zeros([len(self.compromised_drones)*2, 1])
        for name in self.compromised_drones:
            index = self.compromised_drones.index(name)
            self.attack_vector[2*index:2*index+2] = drones[name].ekf.x[0:2] + get_subvector(design_vector.value, index)
