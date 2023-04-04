'''
Oct 22: Joint cyberattack detection and mitigation
---------------------------------------------
Combines LBEKF with distributed mitigation strategy, i.e., offboard detection and onboard mitigation of attacks
'''
import types

import numpy as np
import scipy.linalg as spl

from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from functions import worldtime, sensor, misc, graph, config, mitigation_strategies, plot
from test_lbekf import LBEKF
import subprocess, sys

config.LIMIT_VEL_ACC = True

ANIMATE_ATTACK_SCENARIO = True

# --------------- Sensor Parameters
INIT_VAR = 0.5
EKF_GPS_STD_DEV = 5.0
EKF_RANGE_STD_DEV = 0.1
PROCESS_VAR = 1.0
BIAS_VECTORS = {"10": misc.column([-0.06, -0.00042]), "11": misc.column([-0.00011, -0.008]),
                "4": misc.column([-0.0605, -0.000531]), "5": misc.column([-0.00105, -0.0081])}

bias_start_timestamp = 5.0

# --------------- Simulation Parameters
TIMESTEPS = 600
RESIDUAL_PLOT_XLIM = 700
BIAS_START_TIME = 200
config.PLOT_LIM = 25.8
config.OFFSET = [13.4, 6.9]
config.MARKER_TYPE = 'drone'
config.GPS_TIMEOUT = 1/10.0
config.INS_TIMEOUT = 1/20.0

DETECTION_THRESHOLD_OFFBOARD = 5.0
DETECTION_THRESHOLD_ONBOARD = 2.5
RES_OFFBOARD_LIM = 20.0
RES_ONBOARD_LIM = 6.0

OFFBOARD_SKIP = 4
ONBOARD_SKIP = 1

WIND_DIRECTION = misc.column((0.0001, 0.00025))
WIND_NOISE = 0.00005
# config.WORLD_TIMESTEP = 0.01

# --------------- Network
POSITIONS = [(3.0, 0), (18.0, 3.0), (11.3, 28.2), (26.3, -0.2), (15.0, 13.0),
             (-9.0, 12.1), (-2.9, 19.1), (25.5, 9.1), (34.4, 4.2), (-7.9, -5.9), (30.8, -14.2),
             (0.2, 7.8), (5.6, 16.8), (27.9, 27.2), (15.9, -9.4), (-5.2, 27.6), (35.7, 23.21),
             (4.6, 26.2), (14, 21), (32.7, 11.5), (20.9, 18.3), (30.3, 19.5), (22.7, -12.5),
             (-6.3, 1.5), (-0.1, -7.4), (7.4, -12.8), (19.7, 28.2), (10.9, -1.8), (-2.5, -14.26),
             (34.5, -5.4)]

# --------------------------------------------- Main ----------------------------------------------------------------- #
time = worldtime.time
ZERO_VECTOR = misc.column([0, 0])


def animate_attack_scenario():
    name_list = misc.NameGenerator(name_type='number').generate(len(POSITIONS))
    ordered_positions = sorted(POSITIONS, key=lambda pos: [pos[0], pos[1]])

    G_for_plotting = graph.Graph()
    for i in range(len(name_list)):
        INIT_COV_UAS = [0.9, 0.9, 1e-1, 1e-1]
        G_for_plotting.add_vertex(obj=mitigation_strategies.DroneWithRangeMitigation(name=name_list[i], position=ordered_positions[i],
                                                                        perfect_init_conditions=False, process_noise=[1e-2, 1e-2, 1e-8, 1e-8],
                                                                        poles=[-2.25, -3.55, -2.5, -4], init_cov=INIT_COV_UAS))

    G_for_plotting.set_edges_by_distance(distance=15.0)
    G = deepcopy(G_for_plotting)
    G.remove_edge('5', '10')
    G.remove_edge('10', '14')
    G.remove_edge('10', '15')
    G.remove_edge('10', '11')

    # --------------------------------------------- Main Loop -------------------------------------------------------- #
    def get_fake_gps(obj: sensor.Drone2D):
        return obj.ekf.x[0:2] + misc.white_noise(obj.gps_cov) + BIAS_VECTORS[obj.name]

    def plot_red(self, **kwargs):
        if self.gps_timer.get_elapsed_time() < 0.35:
            plot.plot_point(misc.tuple_from_col_vec(
                self._pos + misc.column(config.GPS_SYMBOL_OFFSET)
            ), color=(0.95, 0.1, 0.2), s=5, edgecolor=(0.85, 0.05, 0.15))

        imgbox = plot.get_image_box("media/lightning.png", zoom=0.255)
        plot.plot_img(misc.tuple_from_col_vec(
            self._pos + misc.column((0.55, 0.9))
        ), imgbox, axis=kwargs.get("axis"), zorder=30)

    # --------------------------------------------- Main Loop -------------------------------------------------------- #
    # config.MARKER_TYPE = "ieee_labeled_graph"
    plot_1 = G_for_plotting.draw()

    for vertex in G.vertices:
        G_for_plotting.vertices[vertex]._pos = G.vertices[vertex]._pos

    def animate(timestep):
        if timestep == BIAS_START_TIME:
            for name in BIAS_VECTORS:
                G.vertices[name].get_gps_measurement = types.MethodType(get_fake_gps, G.vertices[name])
                G_for_plotting.vertices[name].plot = types.MethodType(plot_red, G_for_plotting.vertices[name])
                G.vertices[name].plot = types.MethodType(plot_red, G.vertices[name])

        if BIAS_START_TIME < timestep < BIAS_START_TIME + 100:
            plot_1.cla()
            G_for_plotting.draw(axis=plot_1)
            return

        for _ in range(3):
            for _ in range(2):
                worldtime.step()
                for vertex in G.vertices:
                    G.vertices[vertex].update_physics()

            wind = np.sin(2*np.pi*timestep/TIMESTEPS)*WIND_DIRECTION
            for vertex in G.vertices:
                # Add disturbances
                G.vertices[vertex]._vel += wind + misc.white_noise(cov=[[WIND_NOISE]])
                G.vertices[vertex].update_logic()

        plot_1.cla()
        G_for_plotting.draw(axis=plot_1)

    anim = animation.FuncAnimation(plt.gcf(), animate, tqdm(range(TIMESTEPS)))
    anim.save("media/NovemberMeeting/test_vulnerable_mesh.mp4", fps=30, dpi=200)
    subprocess.call(["open", "media/NovemberMeeting/test_vulnerable_mesh.mp4"])


if __name__ == "__main__":
    if ANIMATE_ATTACK_SCENARIO:
        animate_attack_scenario()
        sys.exit()

    name_list = misc.NameGenerator(name_type='number').generate(len(POSITIONS))
    ordered_positions = sorted(POSITIONS, key=lambda pos: [pos[0], pos[1]])

    G = graph.Graph()
    for i in range(len(name_list)):
        # INIT_COV_UAS = [1e-4, 1e-4, 1e-7, 1e-7]
        INIT_COV_UAS = [0.9, 0.9, 1e-1, 1e-1]
        G.add_vertex(obj=mitigation_strategies.DroneWithRangeMitigation(name=name_list[i], position=ordered_positions[i],
                                        perfect_init_conditions=False, process_noise=[1e-2, 1e-2, 1e-8, 1e-8],
                                        poles=[-2.25, -3.55, -2.5, -4], init_cov=INIT_COV_UAS))

    G.set_edges_by_distance(distance=15.0)
    ekf = LBEKF(network=G, estimate=np.concatenate([misc.column(G.vertices[vtx]._pos)
                                                    + misc.white_noise(cov=np.identity(2) * INIT_VAR)
                                                    for vtx in G.order]),
                beacon_agents=name_list,
                estimate_cov=np.identity(2)*INIT_VAR, process_cov=1.0*PROCESS_VAR*np.identity(2), gps_std_dev=EKF_GPS_STD_DEV,
                range_std_dev=EKF_RANGE_STD_DEV,
                bandwidth=None)

    ekf_without_range = LBEKF(network=G, estimate=np.concatenate([misc.column(G.vertices[vtx]._pos)
                                                    + misc.white_noise(cov=np.identity(2) * INIT_VAR)
                                                    for vtx in G.order]),
                beacon_agents=name_list,
                estimate_cov=np.identity(2)*INIT_VAR, process_cov=1.0*PROCESS_VAR*np.identity(2), gps_std_dev=EKF_GPS_STD_DEV,
                range_std_dev=1000.0*EKF_RANGE_STD_DEV,
                bandwidth=None)

    # --------------------------------------------- Main Loop -------------------------------------------------------- #
    def get_fake_gps(obj: sensor.Drone2D):
        return obj.ekf.x[0:2] + misc.white_noise(obj.gps_cov) + BIAS_VECTORS[obj.name]

    def get_fake_gps_centralized(self, network):
        measurements = np.zeros([self.dim*len(self.beacon_agents) + len(self.edge_list_indices), 1])
        _i = 0
        for name in self.beacon_agents:
            if name in BIAS_VECTORS:
                measurements[_i:_i + self.dim] += (self.get_subvector(network.index(name)) + BIAS_VECTORS[name]
                                                   + misc.white_noise(cov=np.identity(2)*self.gps_std_dev**2))
            else:
                measurements[_i:_i + self.dim] = (network.vertices[name]._pos
                                                  + misc.white_noise(cov=np.identity(2)*self.gps_std_dev**2))
            _i = _i + 2

        for e, e_ind in zip(self.edge_list, self.edge_list_indices):
            measurements[_i][0] = (np.linalg.norm(network.vertices[e[0]]._pos - network.vertices[e[1]]._pos)
                                   + misc.white_noise(cov=[[self.range_std_dev**2]]))
            _i = _i + 1

        return measurements

    def plot_red(self, **kwargs):
        if self.gps_timer.get_elapsed_time() < 0.35:
            plot.plot_point(misc.tuple_from_col_vec(
                self._pos + misc.column(config.GPS_SYMBOL_OFFSET)
            ), color=(0.95, 0.1, 0.2), s=5, edgecolor=(0.85, 0.05, 0.15))

        imgbox = plot.get_image_box("media/lightning.png", zoom=0.255)
        plot.plot_img(misc.tuple_from_col_vec(
            self._pos + misc.column((0.55, 0.9))
        ), imgbox, axis=kwargs.get("axis"), zorder=30)

    # --------------------------------------------- Main Loop -------------------------------------------------------- #

    plot_1 = G.draw()
    onboard_timestamps = []
    offboard_timestamps = []
    offboard_residuals = {vertex: [] for vertex in name_list}
    onboard_residuals = {vertex: [] for vertex in name_list}
    bias_start_timestamp = None
    bias_start_index_onboard = None
    bias_start_index_offboard = None

    def animate(timestep):
        if timestep == BIAS_START_TIME:
            for name in BIAS_VECTORS:
                G.vertices[name].get_gps_measurement = types.MethodType(get_fake_gps, G.vertices[name])
                G.vertices[name].plot = types.MethodType(plot_red, G.vertices[name])
            ekf.get_measurements = types.MethodType(get_fake_gps_centralized, ekf)
            ekf_without_range.get_measurements = types.MethodType(get_fake_gps_centralized, ekf_without_range)
            global bias_start_timestamp
            global bias_start_index_onboard
            global bias_start_index_offboard
            bias_start_timestamp = time()
            bias_start_index_onboard = len(onboard_timestamps)
            bias_start_index_offboard = len(offboard_timestamps)

        for _ in range(3):
            for _ in range(2):
                worldtime.step()
                for vertex in G.vertices.values():
                    vertex.update_physics()

            wind = np.sin(2*np.pi*timestep/TIMESTEPS)*WIND_DIRECTION
            for vertex in G.vertices.values():
                # Add disturbances
                vertex._vel += wind + misc.white_noise(cov=[[WIND_NOISE]])
                vertex.update_logic()
                onboard_residuals[vertex.name].append(float(vertex.gps_residual)*2.0)
            onboard_timestamps.append(time())

        ekf.update_rigidity_matrix(G)
        ekf_without_range.update_rigidity_matrix(G)
        ekf.update(G, ekf.get_measurements(G))
        ekf_without_range.update(G, ekf_without_range.get_measurements(G))
        for vertex in G.vertices:
            _vec = ekf_without_range.get_subvector(G.index(vertex)) - ekf.get_subvector(G.index(vertex))
            offboard_residuals[vertex].append(float(_vec.T @ _vec)/10.0)

        offboard_timestamps.append(time())
        plot_1.cla()
        G.draw(axis=plot_1)

    anim = animation.FuncAnimation(plt.gcf(), animate, tqdm(range(TIMESTEPS)))
    anim.save("media/NovemberMeeting/test_mitigation.mp4", fps=30, dpi=200)
    subprocess.call(["open", "media/NovemberMeeting/test_mitigation.mp4"])

    # ---------------------------------------------------------------------------------------------------------------- #

    for k in name_list:
        offboard_residuals[k] = offboard_residuals[k][::OFFBOARD_SKIP]
    offboard_timestamps = offboard_timestamps[::OFFBOARD_SKIP]
    bias_start_index_offboard = int(bias_start_index_offboard/OFFBOARD_SKIP)

    from matplotlib import cm
    green_cmap = cm.get_cmap('Greens')
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots()
    lines = []

    def float_from_key(_key):
        ind = G.index(_key)
        minimum = 5
        return float(
            (len(G.vertices)-ind+minimum) / (len(G.vertices)+minimum)
        )

    for key in G.vertices:
        line, = ax.plot(offboard_timestamps[:bias_start_index_offboard], offboard_residuals[key][:bias_start_index_offboard], linewidth=1.3, color=green_cmap(float_from_key(key)), alpha=0.7)
        lines.append(line)
    for key in G.vertices:
        color = green_cmap(float_from_key(key)) if key not in BIAS_VECTORS else 'red'
        line, = ax.plot(offboard_timestamps[bias_start_index_offboard:], offboard_residuals[key][bias_start_index_offboard:], linewidth=1.3, color=color, alpha=0.7)
        lines.append(line)

    def update(num, lines):
        for key, line in zip(G.vertices, lines[:len(G.vertices)]):
            line.set_data(offboard_timestamps[:num], offboard_residuals[key][:num])

        for key, line in zip(G.vertices, lines[len(G.vertices):]):
            if key in BIAS_VECTORS:
                line.set_alpha(1.0)
            if num >= bias_start_index_offboard:
                line.set_data(offboard_timestamps[bias_start_index_offboard:num], offboard_residuals[key][bias_start_index_offboard:num])
            else:
                line.set_data([], [])
        return lines

    residual_animation = animation.FuncAnimation(fig, update, fargs=[lines], interval=25, frames=len(offboard_timestamps),
                                                 blit=True)
    ax.plot([offboard_timestamps[0], offboard_timestamps[-1]],
            [DETECTION_THRESHOLD_OFFBOARD, DETECTION_THRESHOLD_OFFBOARD], '--', color='black', linewidth=2.5)

    # ax.plot([bias_start_timestamp, bias_start_timestamp], [0, RES_OFFBOARD_LIM],
    #         '-', color='red',
    #         linewidth=1.0)

    ax.set_ylabel("Residual Values")
    ax.set_xlabel("Time")
    if RES_OFFBOARD_LIM is not None:
        ax.set_ylim([0, RES_OFFBOARD_LIM])
        ax.set_xlim([0, max(offboard_timestamps)*(RESIDUAL_PLOT_XLIM/TIMESTEPS)])

    residual_animation.save('media/NovemberMeeting/test_offboard_residuals.mp4')
    # subprocess.call(["open", 'media/NovemberMeeting/test_offboard_residuals.mp4'])

    # ---------------------------------------------------------------------------------------------------------------- #
    for k in name_list:
        onboard_residuals[k] = onboard_residuals[k][::ONBOARD_SKIP]
    onboard_timestamps = onboard_timestamps[::ONBOARD_SKIP]
    bias_start_index_onboard = int(bias_start_index_onboard/ONBOARD_SKIP)

    for key in ['1', '20', '16', '13', '4', '5', '10', '11']:
        plt.cla()
        plt.clf()
        fig, ax = plt.subplots()
        lines = []
        line, = ax.plot(onboard_timestamps[:bias_start_index_onboard], onboard_residuals[key][:bias_start_index_onboard], linewidth=1.3, color=green_cmap(float_from_key(key)), alpha=0.7)
        lines.append(line)
        color = green_cmap(float_from_key(key)) if key not in BIAS_VECTORS else 'red'
        line, = ax.plot(onboard_timestamps[bias_start_index_onboard:], onboard_residuals[key][bias_start_index_onboard:], linewidth=1.3, color=color, alpha=0.7)
        lines.append(line)

        def update(num, key, lines):
            lines[0].set_data(onboard_timestamps[:num], onboard_residuals[key][:num])
            if key in BIAS_VECTORS:
                lines[1].set_alpha(1.0)
            if num >= bias_start_index_onboard:
                lines[1].set_data(onboard_timestamps[bias_start_index_onboard:num], onboard_residuals[key][bias_start_index_onboard:num])
            else:
                lines[1].set_data([], [])
            return lines

        residual_animation = animation.FuncAnimation(fig, update, fargs=[key, lines], interval=25,
                                                     frames=len(onboard_timestamps), blit=True)
        ax.plot([onboard_timestamps[0], onboard_timestamps[-1]], [DETECTION_THRESHOLD_ONBOARD, DETECTION_THRESHOLD_ONBOARD],
                '--', color='black', linewidth=2.5)

        # ax.plot([bias_start_timestamp, bias_start_timestamp], [0, RES_ONBOARD_LIM],
        #         '-', color='red',
        #         linewidth=1.0)

        ax.set_ylabel("Residual at UAS " + str(key))
        ax.set_xlabel("Time")

        if RES_ONBOARD_LIM is not None:
            ax.set_ylim([0, RES_ONBOARD_LIM])
            ax.set_xlim([0, max(onboard_timestamps)*(RESIDUAL_PLOT_XLIM/TIMESTEPS)])

        residual_animation.save('media/NovemberMeeting/test_onboard_residuals_' + key + '.mp4')
        # subprocess.call(["open", 'media/NovemberMeeting/test_onboard_residuals_' + key + '.mp4'])

