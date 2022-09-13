from functions import sensor, misc, plot, config
import numpy as np
import numpy.linalg as la
from scipy.linalg import block_diag

import functions.worldtime
time = functions.worldtime.time


def get_residual_power(measurement, predicted_measurement, noise_power):
    residual = measurement - predicted_measurement
    return residual.T @  residual / noise_power


class DroneWithRangeMitigation(sensor.Drone2D):
    def __init__(self, *args, if_mitigation=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.clock_2 = misc.Timer()
        self.range_cov = [[0.000001]]
        self.range_timer = misc.Timer(duration=config.GPS_TIMEOUT, randomize=True)
        self.if_mitigation = if_mitigation
        self.gps_residual = 0.0
        self.ins_residual = 0.0
        self.range_residuals = []

    def get_range_measurements(self):
        range_measurements = []
        for neighbor in self.neighbors:
            range_measurements.append([misc.column([neighbor.ekf.x[0][0], neighbor.ekf.x[1][0]]),
                                       la.norm(neighbor._pos - self._pos) + misc.white_noise(self.range_cov)])
        return range_measurements

    def plot(self, **kwargs):
        if self.gps_timer.get_elapsed_time() < 0.35:
            plot.plot_point(misc.tuple_from_col_vec(
                self._pos + misc.column(config.GPS_SYMBOL_OFFSET)
            ), color=(0.05, 0.75, 0.2), s=5, edgecolor=(0.05, 0.85, 0.15))

    def measure(self):
        C_matrix = []
        measurements = []
        noise_covs = []

        if self.gps_timer.get_status_and_reset():
            C_matrix.append(sensor.Drone2D.C_MATRIX_GPS.tolist())
            gps_measurement = self.get_gps_measurement()
            measurements.append(gps_measurement.tolist())
            noise_covs.append(self.gps_cov)
            self.gps_residual = get_residual_power(gps_measurement,
                                                   misc.column([self.ekf.x[0][0], self.ekf.x[1][0]]),
                                                   self.ekf.P[0][0] + self.ekf.P[1][1])

        if self.ins_timer.get_status_and_reset():
            C_matrix.append(sensor.Drone2D.C_MATRIX_INS.tolist())
            ins_measurement = self.get_ins_measurement()
            measurements.append(ins_measurement.tolist())
            noise_covs.append(self.ins_cov)
            self.ins_residual = get_residual_power(ins_measurement,
                                                   misc.column([self.ekf.x[0][0], self.ekf.x[1][0]]),
                                                   self.ekf.P[2][2] + self.ekf.P[3][3])

        if measurements:
            self.ekf.update(C_matrix=np.concatenate(C_matrix),
                            measurement=np.concatenate(measurements),
                            measurement_noise=block_diag(*noise_covs))

        if self.range_timer.get_status_and_reset() and self.if_mitigation:
            range_measurements = self.get_range_measurements()
            self_ekf_pos = misc.column([self.ekf.x[0][0], self.ekf.x[1][0]])
            self_ref_point = misc.column([self.ref_point[0][0], self.ref_point[1][0]])

            k = 0.1
            force = misc.column([0.0, 0.0])
            self.range_residuals = []
            for i in range(len(range_measurements)):
                vec = range_measurements[i][0] - self_ekf_pos
                vec_norm = la.norm(vec)
                self.range_residuals.append(range_measurements[i][1] - vec_norm)
                force -= self.range_residuals[-1] * k * (vec/vec_norm)

            # if functions.worldtime.time() > functions.worldtime.TOTAL_TIME*0.85:
            #     if self.name in ['L', 'A']:
            #         print(self.name, ' is using ', la.norm(force), ' force.')

            self.ekf.x += np.block([[np.identity(2)*0.01], [np.identity(2)]]) @ force

