"""
Sun Sep 25: Plots geometric graphs with the vertices labeled neatly, for publication to IEEE.
"""

import numpy as np
from functions import graph, sensor, misc, config
import matplotlib.pyplot as plt
import numpy as np

cell_size = 2*0.75/3
mesh_y = np.linspace(-1.5*cell_size, 1.5*cell_size, 4)
mesh_x = np.linspace(-2*cell_size, 2*cell_size, 5)
mesh_points = []
for x in mesh_x:
    for y in mesh_y:
        mesh_points.append((x, y))


config.MARKER_TYPE = 'ieee_labeled_graph'
FIGSIZE = [3.5, 2.5]
graphs = {
    "circle": {
        "positions": [(0.75*np.cos(theta + np.pi - 0.0001), 0.75*np.sin(theta + np.pi - 0.0001)) for theta in np.linspace(0, 2*np.pi, 10)][:-1],
        "radius": 0.65,
        "plot_lim_x": [-0.85, 0.85],
        "plot_lim_y": [-0.85, 0.85]
               },
    "mesh": {
        "positions": [(point[0], point[1]) for point in mesh_points],
        "radius": cell_size*1.1,
        "plot_lim_x": [-1.2, 1.2],
        "plot_lim_y": [-0.85, 0.85]
    }
}


def reduced_bandwidth_labeling(positions):
    return sorted(positions, key=lambda pos: [pos[0], pos[1]])


def plot_geometric_graph(positions, radius, plot_lim_x, plot_lim_y):
    # Plot graph within 4x4 centered at the origin.
    G = graph.Graph()
    labels = misc.NameGenerator(name_type='number').generate(len(positions))
    for i in range(len(labels)):
        G.add_vertex(obj=sensor.PhysicalSystem(name=labels[i], position=positions[i]))

    G.set_edges_by_distance(radius)
    G.draw()
    plt.xlim(plot_lim_x)
    plt.ylim(plot_lim_y)


# graph_dict = graphs["circle"]
graph_dict = graphs["mesh"]
plt.figure(figsize=graph_dict.get("figsize", FIGSIZE), dpi=1200)
positions = graph_dict["positions"]
# plot_geometric_graph(positions, graph_dict["radius"])
plot_geometric_graph(reduced_bandwidth_labeling(positions), graph_dict["radius"],
                     graph_dict["plot_lim_x"], graph_dict["plot_lim_y"])
plt.savefig("media/labeled_graph.eps")

import subprocess
subprocess.call(["open", "media/labeled_graph.eps"])
