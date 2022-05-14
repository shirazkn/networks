import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from functions import misc, config


def draw_graph(graph, axis, plot_estimates=False, **kwargs):
    """
    Plots graph with vertices placed in a specific configuration
    Potentially supports 3D configurations(?)
    """
    nx_graph = nx.Graph(graph.edges)
    positions = None
    if graph.vertices[graph.order[0]]._pos is not None:
        positions = {}
        for vertex in graph.vertices.keys():
            positions[vertex] = misc.tuple_from_col_vec(graph.vertices[vertex]._pos)
    nx.draw(nx_graph, positions, with_labels=True, ax=axis, **kwargs)

    for vertex in graph.vertices.values():
        vertex.plot(**kwargs)

    plt.xlim((-config.PLOT_LIM + config.OFFSET[0], config.PLOT_LIM + config.OFFSET[0]))
    plt.ylim((-config.PLOT_LIM + config.OFFSET[1], config.PLOT_LIM + config.OFFSET[1]))
    plt.xticks()
    plt.yticks()
    plt.grid()
    # plt.legend()


def show():
    plt.show()


def get_axis():
    return plt.gca()


def plot_point(point, **style):
    plt.scatter([point[0]], [point[1]], zorder=10, **style)


def plot_line(points, style):
    plt.plot([point[0] for point in points], [point[1] for point in points], style, zorder=-10, linewidth=0.5)
