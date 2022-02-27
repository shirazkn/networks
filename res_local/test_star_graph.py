"""
Sun, Feb 27: Example of how graphs will be represented
Currently uses networkx to plot a graph, ignoring the positions_list
"""

from functions.misc import NameGenerator
from functions import graph, sensor

if __name__ == "__main__":
    name_list = NameGenerator().generate(5)
    positions_list = [(2, 3), (3, 4), (3, 3), (1.5, 4.5), (4.5, 0.5)]
    edges_list = [(0, 2), (1, 2), (3, 2), (4, 2)]
    F = graph.Framework()
    for name, position in zip(name_list, positions_list):
        F.add_vertex(name, sensor.Sensor(), position)

    for edge in edges_list:
        F.add_edge(name_list[edge[0]], name_list[edge[1]])

    F.display()
    