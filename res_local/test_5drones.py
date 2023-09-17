"""
Wed, Mar 23: Example of how a graph vertex can represent a sensor object
Uses a graph with a single vertex, where the vertex represents a mobile sensor
"""
from config import constants
from functions import graph, sensor, misc
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm

from functions import worldtime
import subprocess

constants.PLOT_TRAJECTORIES = True


if __name__ == "__main__":
    name_list = misc.NameGenerator().generate(5)

    G = graph.Graph()
    # Moving drones:
    G.add_vertex(obj=sensor.Drone2D(position=(-5, 0), name=name_list[0],
                                    trajectory=misc.lemniscate_of_bernoulli(a=7, offset=(-5, 0), scale_y=1.2),
                                    perfect_init_conditions=True))
    G.add_vertex(obj=sensor.Drone2D(position=(5, 0), name=name_list[1],
                                    trajectory=misc.lemniscate_of_bernoulli(a=-7, offset=(5, 0), scale_y=1.2,
                                                                               time_offset=0.5),
                                    perfect_init_conditions=True))

    # Stationary drones:
    G.add_vertex(obj=sensor.Drone2D(position=(-8, 0), name=name_list[2],
                                    trajectory=misc.fixed_point_trajectory(point=(-8, 0)),
                                    perfect_init_conditions=True))
    G.add_vertex(obj=sensor.Drone2D(position=(8, 0), name=name_list[3],
                                    trajectory=misc.fixed_point_trajectory(point=(8, 0)),
                                    perfect_init_conditions=True))
    G.add_vertex(obj=sensor.Drone2D(position=(0, -8), name=name_list[4],
                                    trajectory=misc.fixed_point_trajectory(point=(0, -8)),
                                    perfect_init_conditions=True))

    G.add_edge("A", "C")
    G.add_edge("B", "D")
    G.add_edge("C", "D")
    G.add_edge("C", "E")
    G.add_edge("D", "E")

    a = G.draw()

    def animate(i):
        for _ in range(5):
            worldtime.step()
            for vertex in G.vertices.values():
                vertex.update_physics()

        for vertex in G.vertices.values():
            vertex.update_logic()

        a.cla()
        G.draw(axis=a)

    anim = animation.FuncAnimation(plt.gcf(), animate, tqdm(range(200)))
    anim.save("test.mp4", fps=30, dpi=200)
    subprocess.call(["open", "test.mp4"])
