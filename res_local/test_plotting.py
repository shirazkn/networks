"""
Tue May 17th: Example of a GPS spoofing attack and mitigation
"""
import types

from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from functions import worldtime, sensor, misc, graph, config, plot


config.PLOT_LIM = 2.5
config.OFFSET = [1.2, 0.7]
config.MARKER_TYPE = "drone"


if __name__ == "__main__":
    name_list = misc.NameGenerator().generate(2)

    G = graph.Graph()
    positions = [(0.3, 0), (1.0, 0.7)]
    edges = [['A', 'B']]

    for i in range(len(name_list)):
        G.add_vertex(obj=sensor.PhysicalSystem(position=positions[i], name=name_list[i]))

    for e in edges:
        G.add_edge(e[0], e[1])

    G.draw()
    plot_1 = plt.gca()
    G.draw(axis=plot_1)

    def animate(timestep):
        for _ in range(5):
            worldtime.step()
            for vertex in G.vertices.values():
                vertex.update_physics()

        plot_1.cla()
        G.draw(axis=plot_1)

    TIMESTEPS = 40
    worldtime.TOTAL_TIME = TIMESTEPS * config.WORLD_TIMESTEP
    anim = animation.FuncAnimation(plt.gcf(), animate, tqdm(range(TIMESTEPS)))
    anim.save("test.mp4", fps=30, dpi=200)

    import subprocess
    subprocess.call(["open", "test.mp4"])

