import numpy as np
from functions import misc, plot


class Graph:
    """
    Stores a Directed Graph as vertices and edges
    Incidentally this implementation is similar to the one in the <networkx> python package
    """
    def __init__(self, is_undirected=False):
        self.Adj = np.array([[]])
        self.order = []  # ["vertex_1", "vertex_2", ...]  (ordered in the same way as the adjacency matrix)
        self.vertices = {}  # {"vertex_name": <some python object>, ... }
        self.edges = {}  # Directed edges from {"starting vertex": {"ending vertex"}}
        self.nameGenerator = misc.NameGenerator()
        self.is_undirected = is_undirected

    def add_vertex(self, name=None, object=None):
        """Assign name to vertex"""
        if name is None:
            while name not in self.vertices.keys():
                name = self.nameGenerator.new_name()
        elif name in self.vertices.keys():
            raise ValueError("Cannot add duplicate vertex name.")

        self.vertices[name] = object
        self.edges[name] = set()  # No edges added yet..
        self.order.append(name)
        assert(len(self.vertices) == len(self.order))

        self.make_adjacency()  # Adj. gets padded with a row and column of 0s

    def make_adjacency(self):
        self.Adj = np.zeros((len(self.vertices), len(self.vertices)))
        for i, vertex_i in enumerate(self.order):
            for j, vertex_j in enumerate(self.order):
                if vertex_j in self.edges[vertex_i]:
                    self.Adj[i, j] = 1

    def add_edge(self, vertex_1, vertex_2):
        self.edges[vertex_1].add(vertex_2)
        self.Adj[self.index(vertex_1), self.index(vertex_2)] = 1
        if self.is_undirected:
            self.edges[vertex_2].add(vertex_1)
            self.Adj[self.index(vertex_2), self.index(vertex_1)] = 1

    def index(self, vertex):
        """Returns the index of <vertex> in <self.order>"""
        return self.order.index(vertex)


class Framework(Graph):
    """
    Child class of <Graph>, specifies a <position> for each <vertex>
    """
    def __init__(self, is_undirected=True):
        super().__init__(is_undirected=is_undirected)
        self.positions = []  # Stored in the same order as <self.order>

    def add_vertex(self, name=None, object=None, position=None):
        super().add_vertex(name, object)
        self.positions.append(misc.column(position))

    def display(self):
        plot.framework_undirected(self)
