"""
Sun, Feb 27: Example of how graphs can be defined and drawn
"""

from functions.misc import NameGenerator
from functions import graph, sensor
from functions.plot import show

if __name__ == "__main__":
    name_list = NameGenerator().generate(5)
    positions_list = [(2, 3), (3, 4), (3, 3), (1.5, 4.5), (7.5, 0.5)]
    edges_list = [(0, 2), (1, 2), (3, 2), (4, 2)]
    F = graph.Graph()
    for name, position in zip(name_list, positions_list):
        F.add_vertex(sensor.PhysicalSystem(name=name, position=position))

    for edge in edges_list:
        F.add_edge(name_list[edge[0]], name_list[edge[1]])

    F.draw()
    show()
