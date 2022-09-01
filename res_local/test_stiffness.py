"""
Jul 27th: This verifies that the characterization of the stiffness matrix in Zhu et al. (2009-2014?) is erroneous
"""
import pdb

from copy import deepcopy
import networkx as nx
import numpy as np
from functions import worldtime, sensor, misc, graph, config, plot
import scipy as sp

config.PLOT_LIM = 0.2
config.OFFSET = [0.15, 0.13]
config.MARKER_TYPE = "drone"


def get_rigidity_matrix(G):
    nx_G = nx.Graph(G.edges)
    edge_list = list(nx_G.edges)
    R = np.zeros([len(edge_list), 2*len(G.vertices)])
    for i, e in enumerate(edge_list):
        edge_pos = G.vertices[e[0]]._pos.T - G.vertices[e[1]]._pos.T
        R[i, (2*G.index(e[0])):(2*G.index(e[0])+2)] = edge_pos
        R[i, (2*G.index(e[1])):(2*G.index(e[1])+2)] = -1*edge_pos
    return R


def get_incidence_matrix(G):
    nx_G = nx.Graph(G.edges)
    edge_list = list(nx_G.edges)
    I = np.zeros([len(G.vertices), len(edge_list)])
    for i, e in enumerate(edge_list):
        I[G.index(e[0]), i] = 1
        I[G.index(e[1]), i] = -1
    return I


def get_augmented_position_vector(G):
    pos = np.zeros([2*len(G.vertices), 1])
    for v in G.vertices:
        pos[2*G.index(v):(2*G.index(v)+2)] = deepcopy(G.vertices[v]._pos)
    return pos


def get_2D_rotation(angle):
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


positions = [(0.0, 0.13), (0.15, 0.0), (0.3, 0.08), (0.2, 0.2), (0.1, 0.25)]
print(f"Graph has {len(positions)} vertices.")
print(f"Minimally rigid 2D framework should have {2*len(positions) - 3} edges.")

name_list = misc.NameGenerator().generate(len(positions))
minimal_rigid_edges = [['A', 'B'], ['B', 'C'], ['C', 'D'], ['D', 'E'], ['E', 'A'],
                       ['A', 'C'], ['A', 'D']]
extra_edges = [['E', 'B']]

# Make the graphs
minimally_rigid_graph = graph.Graph()
rigid_graph = graph.Graph()
for i in range(len(name_list)):
    minimally_rigid_graph.add_vertex(obj=sensor.PhysicalSystem(position=positions[i], name=name_list[i]))
    rigid_graph.add_vertex(obj=sensor.PhysicalSystem(position=positions[i], name=name_list[i]))

# Add edges
for e in minimal_rigid_edges:
    minimally_rigid_graph.add_edge(e[0], e[1])
    rigid_graph.add_edge(e[0], e[1])
for e in extra_edges:
    rigid_graph.add_edge(e[0], e[1])

# plot_1 = plot.new_figure("Minimally Rigid Graph")
# plot_2 = plot.new_figure("Non-minimally Rigid Graph")
# minimally_rigid_graph.draw(axis=plot_1.gca())
# rigid_graph.draw(axis=plot_2.gca())
# plot.show()

R = get_rigidity_matrix(minimally_rigid_graph)
R_2 = get_rigidity_matrix(rigid_graph)
In = get_incidence_matrix(minimally_rigid_graph)
In_2 = get_incidence_matrix(rigid_graph)
I_V = np.identity(len(rigid_graph.vertices))
I_2 = np.identity(2)

# def animate(timestep):
#     for _ in range(5):
#         worldtime.step()
#         for vertex in G.vertices.values():
#             vertex.update_physics()
#
#     plot_1.cla()
#     G.draw(axis=plot_1)
#
# TIMESTEPS = 40
# worldtime.TOTAL_TIME = TIMESTEPS * config.WORLD_TIMESTEP
# anim = animation.FuncAnimation(plt.gcf(), animate, tqdm(range(TIMESTEPS)))
# anim.save("test.mp4", fps=30, dpi=200)
