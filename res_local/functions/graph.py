import numpy as np
from functions import misc, plot


class Graph:
    """
    Stores a Directed Graph as vertices and edges
    Incidentally this implementation is similar to the one in the <networkx> python package
    """
    def __init__(self, is_undirected=True):
        self.Adj = np.array([[]])
        self.order = []  # ["vertex_1", "vertex_2", ...]  (ordered in the same way as the adjacency matrix)
        self.vertices = {}  # {"vertex_name": <some python object>, ... }
        self.edges = {}  # Directed edges from {"starting vertex": {"ending vertex"}}
        self.nameGenerator = misc.NameGenerator()
        self.is_undirected = is_undirected

    def add_vertex(self, obj=None):
        """Assign name to vertex"""
        if obj.name is None:
            while obj.name not in self.vertices.keys():
                obj.name = self.nameGenerator.new_name()
        elif obj.name in self.vertices.keys():
            raise ValueError("Cannot add duplicate vertex name.")

        self.vertices[obj.name] = obj
        self.edges[obj.name] = set()  # No edges added yet..
        self.order.append(obj.name)
        assert(len(self.vertices) == len(self.order))

    def make_adjacency(self):
        self.Adj = np.zeros((len(self.vertices), len(self.vertices)))
        for i, vertex_i in enumerate(self.order):
            for j, vertex_j in enumerate(self.order):
                if vertex_j in self.edges[vertex_i]:
                    self.Adj[i, j] = 1

    def add_edge(self, vertex_1, vertex_2):
        self.edges[vertex_1].add(vertex_2)
        self.vertices[vertex_1].neighbors.add(self.vertices[vertex_2])
        if self.is_undirected:
            self.edges[vertex_2].add(vertex_1)
            self.vertices[vertex_2].neighbors.add(self.vertices[vertex_1])

    def set_edges_by_distance(self, distance):
        for key in self.edges:
            self.edges[key] = set()

        for i in range(len(self.vertices)):
            v_1 = self.order[i]
            for v_2 in self.order[i+1:]:
                if np.linalg.norm(self.vertices[v_1]._pos - self.vertices[v_2]._pos) < distance:
                    self.add_edge(v_1, v_2)

    def get_objects(self, keys: list):
        return_list = []
        for key in keys:
            return_list.append(self.vertices[key])
        return return_list

    def index(self, vertex):
        """Returns the index of <vertex> in <self.order>"""
        return self.order.index(vertex)

    def draw(self, axis=None, title=None, **kwargs):
        plot.draw_graph(self, axis, title, **kwargs)
        return plot.get_axis() if not axis else axis

    def get_directed_edge_list(self):
        edge_list = []
        for starting_vertex in self.edges.keys():
            for ending_vertex in self.edges[starting_vertex]:
                edge_list.append((starting_vertex, ending_vertex))
        return edge_list

    def show(self):
        plot.show()

    # TODO: Implement 'check_connectivity()' to connect nodes based on Euclidean distance
