import types
import cvxpy as cvx
import dccp
from copy import deepcopy
import numpy as np
import numpy.linalg as la
from scipy.linalg import block_diag
from functions import misc, plot, mitigation_strategies, config


def plot_red(self, **kwargs):
    if self.gps_timer.get_elapsed_time() < 0.35:
        plot.plot_point(misc.tuple_from_col_vec(
            self._pos + misc.column(config.GPS_SYMBOL_OFFSET)
        ), color=(0.95, 0.1, 0.2), s=5, edgecolor=(0.85, 0.05, 0.15))

    imgbox = plot.get_image_box("media/lightning.png", zoom=0.275)
    plot.plot_img(misc.tuple_from_col_vec(
        self._pos + misc.column((0.17, 0.22))
                                          ), imgbox, axis=kwargs.get("axis"), zorder=30)


def get_subvector(vec, index):
    # Gets index-th column subvector of length 2 from a column vector
    return vec[2*index:(2*index)+2]


class Attacker:
    def __init__(self):
        self.compromised_drones = []
        self.attack_vectors = []
        self.attack_power_limit = 0.1

        # Choose Attack Type
        # self.design_attack = self.design_optimal_attack
        self.design_attack = self.design_greedy_attack

    def add_compromised_drone(self, drone):
        self.compromised_drones.append(drone.name)

        # Whenever drone calls get_gps, call get_fake_gps instead:
        drone.get_gps_measurement = types.MethodType(self.get_fake_gps, drone)

        # For plotting purposes:
        drone.plot = types.MethodType(plot_red, drone)

    def get_fake_gps(self, obj: mitigation_strategies.DroneWithRangeMitigation):
        # Must call design_attack first!
        index = self.compromised_drones.index(obj.name)
        return obj.ekf.x[0:2] + misc.white_noise(obj.gps_cov) + self.attack_vectors[index]

    def design_optimal_attack(self, drones: [mitigation_strategies.DroneWithRangeMitigation]):
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
        P = temp_drone.ekf.P
        measurement_noise = block_diag(temp_drone.gps_cov, temp_drone.ins_cov)
        A = np.block([[I_2, I_2*self.prediction_timestep], [np.zeros([2, 2]), I_2]])
        K = A @ P @ np.linalg.pinv((measurement_noise + P))

        M_1 = I_2*self.prediction_timestep - K[0:2, 2:4]
        M_2 = I_2 - K[2:4, 2:4]
        K_P = K[0:2, 0:2]
        K_VP = K[2:4, 0:2]
        M_3 = A - K

        constraints = []
        for _ in range(1, self.optimization_window+1):
            for name in drones:
                if name in self.compromised_drones:
                    del_a = get_subvector(design_vector, self.compromised_drones.index(name))
                    position_errors[name] = position_errors[name] + M_1 @ velocity_errors[name] + K_P @ del_a
                    velocity_errors[name] = M_2 @ velocity_errors[name] + K_VP @ del_a

                    position_res = position_errors[name] + del_a
                    velocity_res = velocity_errors[name]
                    constraints.append((cvx.norm(position_res)) <= self.gps_residual_limit)
                    constraints.append((cvx.norm(velocity_res)) <= self.ins_residual_limit)

                    for neighbor in drones[name].neighbors:
                        del_e = (position_errors[neighbor.name] - position_errors[name])
                        constraints.append((del_e.T @ (neighbor._pos - drones[name]._pos))
                                           <= self.range_residual_limit)
                        constraints.append((-1*del_e.T @ (neighbor._pos - drones[name]._pos))
                                           <= self.range_residual_limit)
                        # soft_constraints += cvx.abs(del_e.T @ (neighbor._pos - drones[name]._pos))

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

    def design_greedy_attack(self, drones: {mitigation_strategies.DroneWithRangeMitigation}):
        if not self.compromised_drones:
            return

        R = []  # Transpose of rigidity matrix
        constraints = []
        design_vector = cvx.Variable((len(self.compromised_drones)*2, 1))

        for i, c_drone in enumerate(self.compromised_drones):
            for neighbor in drones[c_drone].neighbors:
                if neighbor in self.compromised_drones:
                    j = self.compromised_drones.index(neighbor.name)
                    if j > i:
                        del_p = (drones[c_drone]._pos - neighbor._pos).T[0]
                        R.append(np.zeros((len(self.compromised_drones)*2)))
                        R[-1][i*2:(i+1)*2] = del_p
                        R[-1][j*2:(j+1)*2] = -1*del_p
                    continue

                del_p = (drones[c_drone]._pos - neighbor._pos).T[0]
                R.append(np.zeros((len(self.compromised_drones)*2)))
                R[-1][i*2:(i+1)*2] = del_p

        for i in range(len(R)):
            R.append(-1*R[i])
        R = np.array(R)

        constraints += [cvx.norm(design_vector) == 0.2]
        optimization_problem = cvx.Problem(cvx.Minimize(cvx.max(R @ design_vector)),
                                           constraints)
        optimization_problem.solve(method='dccp')

        self.attack_vectors = []
        max_attack_norm = 0.0
        for i in range(len(self.compromised_drones)):
            self.attack_vectors.append(get_subvector(design_vector.value, i))
            max_attack_norm = max(max_attack_norm, la.norm(self.attack_vectors[-1]))

        # for vec in self.attack_vectors:
        #     vec *= self.attack_power_limit/max_attack_norm

        intent = misc.column((-1.0, +0.5))
        sign_flip = np.sign(intent.T @ self.attack_vectors[0])
        for i in range(len(self.compromised_drones)):
            self.attack_vectors[i] *= sign_flip*self.attack_power_limit/max_attack_norm

        return
