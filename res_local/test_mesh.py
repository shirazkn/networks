"""
Mon, Apr 4: Example of a mesh network
"""

from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm

from functions import worldtime, sensor, misc, graph, config


config.PLOT_LIM = 2.5
config.OFFSET = [1.2, 0.7]

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
                     object=sensor.Drone2D(position=positions[i],
                                           trajectory=misc.fixed_point_trajectory(point=positions[i]),
                                           perfect_init_conditions=True, process_noise=[1e-7, 1e-7, 1e-8, 1e-8],
                                           poles=[-1, -0.75, -0.5, -2], init_cov=[1e-2, 1e-2, 1e-3, 1e-3]))
    for e in edges:
        G.add_edge(e[0], e[1])

    a = G.draw()

    def animate(_):
        for _ in range(5):
            worldtime.step()
            for vertex in G.vertices.values():
                vertex.update_physics()

        for vertex in G.vertices.values():
            vertex.update_logic()

        a.cla()
        G.draw(axis=a)

    anim = animation.FuncAnimation(plt.gcf(), animate, tqdm(range(20)))
    anim.save("test.mp4", fps=30, dpi=200)
