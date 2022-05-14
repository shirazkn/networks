"""
Mon, Apr 4: Example of a mesh network
"""

from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from functions import worldtime, sensor, misc, graph, config, plot

import numpy as np
import numpy.linalg


config.PLOT_LIM = 2.5
config.OFFSET = [1.2, 0.7]


class Drone2D(sensor.Drone2D):
    def __init__(self, *args, mitigation=True, **kwargs):
        self.clock_2 = misc.Timer()
        self.mitigation = mitigation
        super().__init__(*args, **kwargs)

    def incorporate_distance_measurements(self, dt):
        measurements = []
        for neighbor in self.neighbors:
            measurements.append([misc.column([neighbor.ekf.x[0][0], neighbor.ekf.x[1][0]]),
                                 numpy.linalg.norm(neighbor._pos - self._pos) + misc.white_noise([[0.1]])])

        if self.mitigation:
            k = 0.1
            force = misc.column([0.0, 0.0])
            for m in measurements:
                vec = m[0] - misc.column([self.ekf.x[0][0], self.ekf.x[1][0]])
                vec_norm = np.linalg.norm(vec)
                # Proj = vec @ vec.T / (vec.T @ vec)
                force += (m[1] - np.linalg.norm(vec)) * k * (vec/vec_norm)

            self.ekf.x -= 0.5*numpy.block([[np.zeros([2, 2])], [np.identity(2)]]) @ force * dt * dt
        return

    def update_logic(self):
        super().update_logic()
        dt = self.clock_2.get_time_and_reset()
        self.incorporate_distance_measurements(dt)

    def plot(self, **kwargs):
        if self.gps_timer.get_elapsed_time() < 0.1:
            plot.plot_point(misc.tuple_from_col_vec(
                self._pos + misc.column([0.1, 0.1])
            ), color=(0.05, 0.75, 0.2), s=30, edgecolor=(0.05, 0.85, 0.15))


class Drone2DSpecial(Drone2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_gps_measurement(self):
        return self._pos + misc.white_noise(self.gps_cov) + misc.column((0.5, -0.7))

    def plot(self, **kwargs):
        if self.gps_timer.get_elapsed_time() < 0.1:
            plot.plot_point(misc.tuple_from_col_vec(
                self._pos + misc.column([0.1, 0.1])
            ), color=(0.95, 0.1, 0.2), s=30, edgecolor=(0.85, 0.05, 0.15))


if __name__ == "__main__":
    name_list = misc.NameGenerator().generate(13)

    G = graph.Graph()
    positions = [(0.3, 0), (1.0, 0.7), (2.2, 0.5), (2.6, -0.6), (2.2, 1.5),  # A..E
                 (-0.7, 0.4), (-0.3, 2.1), (3.0, 0.7),  # ..H
                 (-0.1, 1.1), (0.6, 1.6), (1.5, -0.5),  # ..K
                 (0.7, 2.3), (1.5, 2.4)]  # ..M
    edges = [['A', 'F'], ['A', 'I'], ['A', 'B'], ['G', 'I'], ['G', 'J'], ['G', 'I'], ['J', 'I'], ['J', 'B'],
             ['B', 'C'], ['B', 'E'], ['B', 'K'], ['C', 'E'], ['C', 'H'], ['C', 'D'], ['C', 'K'], ['K', 'D'],
             ['I', 'F'], ['I', 'F'], ['D', 'H'], ['J', 'L'], ['L', 'M'], ['M', 'E']]

    # Stationary drones:
    for i in range(13):
        G.add_vertex(name=name_list[i],
                     object=Drone2D(position=positions[i],
                                    trajectory=misc.fixed_point_trajectory(point=positions[i]),
                                    perfect_init_conditions=True, process_noise=[1e-7, 1e-7, 1e-8, 1e-8],
                                    poles=[-5, -5.75, -4.5, -4], init_cov=[1e-4, 1e-4, 1e-7, 1e-7]))

    G.vertices["J"] = Drone2DSpecial(position=positions[9],
                                     trajectory=misc.fixed_point_trajectory(point=positions[9]),
                                     perfect_init_conditions=True, process_noise=[1e-7, 1e-7, 1e-8, 1e-8],
                                     poles=[-5, -5.75, -4.5, -4], init_cov=[1e-4, 1e-4, 1e-7, 1e-7])

    for e in edges:
        G.add_edge(e[0], e[1])

    G_a = deepcopy(G)

    for v in G.vertices.values():
        v.mitigation = False

    for v in G_a.vertices.values():
        v.mitigation = True

    plot_1 = G.draw()
    G_a.draw(axis=plot_1)

    def animate(_):
        for _ in range(5):
            worldtime.step()
            for vertex in G.vertices.values():
                vertex.update_physics()
            for vertex in G_a.vertices.values():
                vertex.update_physics()

        for vertex in G.vertices.values():
            vertex.update_logic()
        for vertex in G_a.vertices.values():
            vertex.update_logic()

        plot_1.cla()
        G.draw(axis=plot_1, alpha=0.2)
        G_a.draw(axis=plot_1)

    anim = animation.FuncAnimation(plt.gcf(), animate, tqdm(range(300)))
    anim.save("test.mp4", fps=30, dpi=200)
