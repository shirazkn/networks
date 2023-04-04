import scipy.linalg
from scipy.signal import place_poles
from functions import misc, plot, config

import numpy as np
from scipy import linalg

import functions.worldtime
time = functions.worldtime.time

# Note: These imports might be useful for auto differentiation (not implemented),
#   from jax import grad
#   import jax.numpy as jnp
#   from functools import partial


class ExtendedKalmanFilter:
    def __init__(self, init_est, init_cov):
        self.x = init_est
        self.dim = len(init_est)
        self.P = init_cov

    def update(self, C_matrix, measurement, measurement_noise):
        K = self.P @ C_matrix.T @ np.linalg.inv((measurement_noise + C_matrix @ self.P @ C_matrix.T))
        self.x = self.x + K@(measurement - C_matrix @ self.x)
        self.P = (np.identity(self.dim) - K @ C_matrix) @ self.P

    def propagate(self, linear_model, process_noise_cov, input):
        self.x = linear_model@self.x + input
        self.P = linear_model@self.P@linear_model.T + process_noise_cov


class PhysicalSystem:
    def __init__(self, position=None, name=None):
        self.name = name
        self._pos = misc.column(position)
        self.dimensions = len(position) if position else None
        if self.dimensions:
            self._vel = misc.column([0.0 for _ in range(self.dimensions)])
            self._acc = misc.column([0.0 for _ in range(self.dimensions)])
        self._simulation_clock = misc.Timer()
        self.neighbors = set()
        self._pos_list = []

    def plot(self, **kwargs):
        if config.PLOT_TRAJECTORIES:
            plot.plot_line(self._pos_list, 'r-')

    def update_physics(self):
        """
        Uses values which were set during the previous logic update
        """
        if config.LIMIT_VEL_ACC:
            speed = np.linalg.norm(self._vel)
            acc = np.linalg.norm(self._acc)
            SPEED_LIMIT = 1.5
            ACC_LIMIT = 0.25
            if speed > SPEED_LIMIT:
                self._vel = SPEED_LIMIT*self._vel/speed
            if acc > ACC_LIMIT:
                self._acc = ACC_LIMIT*self._acc/acc

        for _ in range(int(self._simulation_clock.get_time_and_reset() / config.ODE_TIMESTEP)):
            self._pos += self._vel*config.ODE_TIMESTEP
            self._vel += self._acc*config.ODE_TIMESTEP
            self._pos_list.append(self._pos.tolist())

    def update_logic(self):
        raise NotImplementedError("Must be defined in child class")


class Drone2D(PhysicalSystem):
    """
    Wireless sensor that moves around and uses an EKF for closed-loop tracking
    """
    # State vector = [pos_x, pos_y, vel_x, vel_y]
    A_MATRIX = np.block([[np.zeros([2, 2]), np.identity(2)],
                         [np.zeros([2, 2]), np.zeros([2, 2])]])
    B_MATRIX = np.block([[np.zeros([2, 2])],
                         [np.identity(2)]])
    C_MATRIX_GPS = np.identity(4)[0:2][:]
    C_MATRIX_INS = np.identity(4)[2:4][:]

    def __init__(self, ins_var=0.0001, gps_var=0.001, trajectory=None, perfect_init_conditions=True,
                 process_noise=[0.01, 0.01, 0.01, 0.01], poles=[-2, -4, -2, -3], init_cov=[1.0, 1.0, 1.0, 1.0],
                 **kwargs):
        super().__init__(**kwargs)
        self.ekf = ExtendedKalmanFilter(init_est=np.zeros([4, 1]),
                                        init_cov=np.diag(init_cov))

        self.process_noise_cov = np.diag(process_noise)

        self.ins_cov = np.diag([ins_var for _ in range(2)])
        self.ins_timer = misc.Timer(duration=config.INS_TIMEOUT, randomize=True)

        self.gps_cov = np.diag([gps_var for _ in range(2)])
        self.gps_timer = misc.Timer(duration=config.GPS_TIMEOUT, randomize=True)

        self.ref_trajectory = trajectory if trajectory else misc.fixed_point_trajectory(self._pos.T[0])
        self.ref_point = misc.column(self.ref_trajectory(time()))
        self.clock = misc.Timer()

        if perfect_init_conditions:
            self._pos = self.ref_point[0:2]
            self._vel = self.ref_point[2:4]
            self.ekf.x = self.ref_point
        else:
            self._pos = self.ref_point[0:2]
            self._vel = self.ref_point[2:4]
            self.ekf.x = self.ref_point + 0.1*Drone2D.C_MATRIX_GPS.T @ misc.white_noise(self.gps_cov)

        # Pole placement of (A + BK)
        self.K = -1.0*place_poles(Drone2D.A_MATRIX, Drone2D.B_MATRIX, poles).gain_matrix

    def get_ins_measurement(self):
        return self._vel + misc.white_noise(self.ins_cov)

    def get_gps_measurement(self):
        return self._pos + misc.white_noise(self.gps_cov)

    def measure(self):
        C_matrix = []
        measurement = []
        noise_covs = []

        if self.gps_timer.get_status_and_reset():
            C_matrix.append(Drone2D.C_MATRIX_GPS.tolist())
            measurement.append(self.get_gps_measurement().tolist())
            noise_covs.append(self.gps_cov)

        if self.ins_timer.get_status_and_reset():
            C_matrix.append(Drone2D.C_MATRIX_INS.tolist())
            measurement.append(self.get_ins_measurement().tolist())
            noise_covs.append(self.ins_cov)

        if measurement:
            self.ekf.update(C_matrix=np.concatenate(C_matrix),
                            measurement=np.concatenate(measurement),
                            measurement_noise=linalg.block_diag(*noise_covs))

    def update_logic(self):
        """
        This function does simple closed-loop trajectory tracking using the ekf estimate
        In a real world, the drone can only influence its current acceleration, so this function:
            1: sets the current acceleration setting (Zero-order hold)
            2: propagates the ekf estimate from current time-step to next time-step
        """
        self.measure()

        self.ref_point = misc.column(self.ref_trajectory(time()))
        error = self.ekf.x - self.ref_point
        self._acc = self.K @ error

        dt = self.clock.get_time_and_reset()
        self.ekf.propagate(linear_model=np.identity(4) + Drone2D.A_MATRIX*dt,
                           process_noise_cov=self.process_noise_cov,
                           input=dt*Drone2D.B_MATRIX@self._acc)

    def plot(self, **kwargs):
        # if self.gps_timer.get_elapsed_time() < 0.15:
        #     plot.plot_point(misc.tuple_from_col_vec(
        #         self._pos + misc.column([0.1, 0.1])
        #     ), color=(0.1, 0.85, 0.2), s=30, edgecolor=(0.85, 0.65, 0.85))
        # super().plot()
        pass
