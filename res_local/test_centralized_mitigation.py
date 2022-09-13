"""
Thu Sep 1: Uses the CLBIF (Centralized L-Banded Information Filter) of Moura, et al.
combined with the vertex relabeling approach I developed
"""
import types

import numpy as np
import scipy.linalg as spl

from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from functions import worldtime, sensor, misc, graph, config, plot, attacker

time = worldtime.time

BIAS_START_TIME = 100
BIAS_VECTORS = {"L": misc.column([-0.06, 0.008]), "M": misc.column([-0.061, 0.001])}
ZERO_VECTOR = misc.column([0, 0])
TIMESTEPS = 400
EXTRA_EDGES = False
DETECTION_THRESHOLD = 0.08
YLIM = 0.14

config.PLOT_LIM = 2.3
config.OFFSET = [1.2, 0.9]
config.MARKER_TYPE = 'drone'


class CLBIF:
    def __init__(self, num_agents, network, estimate, estimate_cov, process_cov, gps_std_dev=0.01, range_std_dev=0.1):
        for i, agent in enumerate(network.vertices):
            assert i == network.index(agent)

        self.num_agents = num_agents
        self.pos = deepcopy(estimate)
        self.vel = np.zeros_like(self.pos)
        self.dim = 2

        self.process_cov = spl.block_diag(*[process_cov for _ in range(num_agents)])
        self.post_cov = spl.block_diag(*[estimate_cov for _ in range(num_agents)])

        self.prior_cov = self.post_cov + self.process_cov

        self.gps_std_dev = gps_std_dev
        self.range_std_dev = range_std_dev

        self.edge_list = network.get_directed_edge_list()
        self.edge_list_indices = []
        for edge in self.edge_list:
            self.edge_list_indices.append((network.index(edge[0]), network.index(edge[1])))

        self.Rig = np.zeros([len(self.edge_list_indices), 2*self.num_agents])
        self.K = np.zeros([self.dim * self.num_agents, self.dim*self.num_agents + len(self.edge_list_indices)])
        self.bias = False

    def update_rigidity_matrix(self, G):
        for i, e in enumerate(self.edge_list_indices):
            edge_pos = self.get_subvector(e[0]) - self.get_subvector(e[1])
            edge_length = np.linalg.norm(edge_pos)
            self.Rig[i, (self.dim*e[0])
                        :((self.dim*e[0])+self.dim)] = (edge_pos.T / (edge_length * self.range_std_dev))
            self.Rig[i, (self.dim*e[1])
                        :((self.dim*e[1])+self.dim)] = (-1*edge_pos.T / (edge_length * self.range_std_dev))

    def get_subvector(self, i):
        return self.pos[self.dim * i : self.dim * (i+1)]

    def update(self, network: graph.Graph):
        innovations = self.update_measurements(network)

        self.update_rigidity_matrix(network)
        meas_inf = (1/self.gps_std_dev**2) * np.identity(self.dim*self.num_agents) + self.Rig.T @ self.Rig
        self.post_cov = np.linalg.inv(np.linalg.inv(self.prior_cov) + meas_inf)
        self.prior_cov = self.post_cov + self.process_cov

        self.K[:, :] = self.post_cov @ (np.concatenate([(1/self.gps_std_dev**2) * np.identity(self.dim*self.num_agents),
                                                        (1/self.range_std_dev) * self.Rig])).T
        # import pdb; pdb.set_trace()
        self.pos = self.pos + self.vel + (self.K @ innovations)

    def update_measurements(self, network):
        """
        Also updates self.vel
        """
        innovations = np.zeros([self.dim*self.num_agents + len(self.edge_list_indices), 1])
        _i = 0
        for name in network.order:
            agent = network.vertices[name]
            self.vel[_i:_i + self.dim] = agent._vel
            innovations[_i:_i + self.dim] = (agent._pos
                                             + misc.white_noise(cov=np.identity(2)*self.gps_std_dev**2)
                                             - self.get_subvector(int(_i/2)))
            if self.bias and name in BIAS_VECTORS:
                innovations[_i:_i + self.dim] += (-1)*agent._pos + self.get_subvector(int(_i/2)) \
                                                 + BIAS_VECTORS[name]
            _i = _i + 2

        for e, e_ind in zip(self.edge_list, self.edge_list_indices):
            innovations[_i][0] = (np.linalg.norm(network.vertices[e[0]]._pos - network.vertices[e[1]]._pos)
                               + misc.white_noise(cov=[[self.range_std_dev**2]])
                               - np.linalg.norm(self.get_subvector(e_ind[0]) - self.get_subvector(e_ind[1])))
            _i = _i + 1

        return innovations

    def plot(self, axis):
        # for i in range(self.num_agents):
        #     plt.Circle(self.get_subvector(i), 0.5, color='b', fill=False)
        pass


if __name__ == "__main__":
    name_list = misc.NameGenerator().generate(15)

    G = graph.Graph()
    positions = [(0.3, 0), (1.0, 0.7), (1.8, 0.3), (2.6, -0.6), (1.5, 1.3),  # A..E
                 (-0.7, 0.4), (-0.3, 2.1), (3.0, 0.7),  # ..H
                 (-0.1, 1.1), (0.5, 1.5), (1.5, -0.5),  # ..K
                 (0.5, 2.5), (1.4, 2.1), (2.5, 1.35), (2.7, 1.9)]  # ..N
    edges = [['A', 'F'], ['A', 'I'], ['A', 'B'], ['G', 'I'], ['G', 'J'], ['G', 'I'], ['J', 'I'], ['J', 'B'],
             ['B', 'C'], ['B', 'E'], ['B', 'K'], ['C', 'E'], ['C', 'H'], ['C', 'D'], ['C', 'K'], ['K', 'D'],
             ['I', 'F'], ['I', 'F'], ['D', 'H'], ['J', 'L'], ['L', 'M'], ['M', 'E'], ['N', 'E'], ['N', 'H'],
             ['N', 'O'], ['O', 'H']]

    for i in range(len(name_list)):
        G.add_vertex(obj=sensor.Drone2D(name=name_list[i], position=positions[i],
                                        trajectory=misc.fixed_point_trajectory(point=positions[i]),
                                        perfect_init_conditions=False, process_noise=[1e-8, 1e-8, 1e-8, 1e-8],
                                        poles=[-2, -3.75, -5.5, -6], init_cov=[1e-4, 1e-4, 1e-7, 1e-7]))

    for e in edges:
        G.add_edge(e[0], e[1])

    if EXTRA_EDGES:
        G.add_edge('L', 'E')
        G.add_edge('J', 'M')

    def get_fake_gps(obj: sensor.Drone2D):
        return obj.ekf.x[0:2] + misc.white_noise(obj.gps_cov) + BIAS_VECTORS[obj.name]

    plot_1 = G.draw()

    gcs = CLBIF(15, network=G, estimate=np.concatenate([misc.column(vec) + misc.white_noise(cov=np.identity(2) * 0.001)
                                                        for vec in positions]),
                estimate_cov=np.identity(2)*0.01, process_cov=np.identity(2), gps_std_dev=0.01, range_std_dev=0.0005)

    gcs_no_range = CLBIF(15, network=G, estimate=np.concatenate([misc.column(vec) + misc.white_noise(cov=np.identity(2)*0.001)
                                                        for vec in positions]),
                estimate_cov=np.identity(2)*0.01, process_cov=np.identity(2), gps_std_dev=10.0, range_std_dev=1000.0)

    residuals = {vertex: [] for vertex in G.vertices.keys()}
    timestamps = []

    def animate(timestep):
        if timestep == BIAS_START_TIME:
            # self.compromised_drones.append(drone.name)
            # Whenever drone calls get_gps, call get_fake_gps instead:
            for drone in (G.vertices['L'], G.vertices['M']):
                drone.get_gps_measurement = types.MethodType(get_fake_gps, drone)
                drone.plot = types.MethodType(attacker.plot_red, drone)

            gcs.bias = True
            gcs_no_range.bias = True

        for _ in range(5):
            worldtime.step()
            for vertex in G.vertices.values():
                vertex.update_physics()

        for vertex in G.vertices.values():
            vertex.update_logic()

        gcs.update(G)
        gcs_no_range.update(G)
        timestamps.append(time())
        for vertex in G.vertices.keys():
            residuals[vertex].append(np.linalg.norm(G.vertices[vertex].ekf.x[0:2] - gcs.get_subvector(G.index(vertex))))
            # residuals[vertex].append(np.linalg.norm(gcs_no_range.get_subvector(G.index(vertex))
            #                                         - gcs.get_subvector(G.index(vertex))))

        plot_1.cla()
        G.draw(axis=plot_1)

    worldtime.TOTAL_TIME = TIMESTEPS * config.WORLD_TIMESTEP
    anim = animation.FuncAnimation(plt.gcf(), animate, tqdm(range(TIMESTEPS)))
    anim.save("test.mp4", fps=30, dpi=200)

    plt.cla()
    plt.clf()
    fig, ax = plt.subplots()
    # ax.set_xlim(9, 0)
    lines = []
    for key in G.vertices:
        # line, = ax.plot(timestamps, residuals[key], label="UAV " + key)
        line, = ax.plot(timestamps, residuals[key], linewidth=1.5)
        lines.append(line)

    def update(num, lines):
        for key, line in zip(G.vertices, lines):
            line.set_data(timestamps[:num], residuals[key][:num])
        return lines

    residual_animation = animation.FuncAnimation(fig, update, fargs=[lines], interval=25, frames=len(timestamps),
                                                 blit=True)
    ax.plot([timestamps[0], timestamps[-1]], [DETECTION_THRESHOLD, DETECTION_THRESHOLD], '--', color='red', label="Detection Threshold",
            linewidth=2.5)
    ax.set_ylabel("Residual Values")
    ax.set_xlabel("Time")
    if YLIM:
        ax.set_ylim([0, YLIM])
    ax.legend()
    residual_animation.save('test_residual.mp4')

    import subprocess
    subprocess.call(["open", "test_residual.mp4"])
    subprocess.call(["open", "test.mp4"])

"""
Animating a Line
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
line, = ax.plot(x, y, color='k')

def update(num, x, y, line):
    line.set_data(x[:num], y[:num])
    line.axes.axis([0, 10, 0, 1])
    return line,

ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line],
                              interval=25, blit=True)
ani.save('test.gif')
plt.show()
"""