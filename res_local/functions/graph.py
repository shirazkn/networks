import numpy as np
from functions import misc, plot
import networkx as nx

nxGraph = nx.Graph

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

    def remove_edge(self, vertex_1, vertex_2):
        self.edges[vertex_1].remove(vertex_2)
        self.vertices[vertex_1].neighbors.remove(self.vertices[vertex_2])
        if self.is_undirected:
            self.edges[vertex_2].remove(vertex_1)
            self.vertices[vertex_2].neighbors.remove(self.vertices[vertex_1])

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
        if list(self.vertices.values())[0].dimensions == 3:
            axis = self.draw3D(axis, title, **kwargs)
        else:
            plot.draw_graph(self, axis, title, **kwargs)
        return plot.get_axis() if not axis else axis

    def draw3D(self, axis, title, **kwargs):
        label_nodes = kwargs.pop("label_nodes") if "label_nodes" in kwargs else True
        zorder = kwargs.pop("zorder") if "zorder" in kwargs else None
        
        node_size = kwargs.pop("node_size") if "node_size" in kwargs else 40
        node_color = kwargs.pop("node_color") if "node_color" in kwargs else "black"
        node_alpha = kwargs.pop("node_alpha") if "node_alpha" in kwargs else 1.0

        edge_width = kwargs.pop("edge_width") if "edge_width" in kwargs else 0.25
        edge_color = kwargs.pop("edge_color") if "edge_color" in kwargs else "black"
        edge_alpha = kwargs.pop("edge_alpha") if "edge_alpha" in kwargs else 1.0

        node_xyz = np.array([misc.tuple_from_col_vec(self.vertices[v]._pos) for v in self.order])
        edge_tuples = []
        for i in range(len(self.order)):
            for j in range(i, len(self.order)):
                if self.order[j] in self.edges[self.order[i]]:
                    edge_tuples.append((misc.tuple_from_col_vec(self.vertices[self.order[i]]._pos), 
                                        misc.tuple_from_col_vec(self.vertices[self.order[j]]._pos)))
        edge_xyz = np.array(edge_tuples)
        if not axis:
            fig = plot.plt.figure()
            ax = fig.add_subplot(111, projection="3d", 
                                 computed_zorder=False if zorder else True)
            fig.tight_layout()
        else:
            ax = axis

        # Alpha is scaled by 'depth'
        ax.scatter(*node_xyz.T, s=node_size, color=node_color, alpha=node_alpha, zorder=zorder, **kwargs)
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, zorder=zorder, linewidth=edge_width, color=edge_color, alpha=edge_alpha, **kwargs)

        ax.grid(False)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        if label_nodes:
            for v in self.vertices:
                plot.annotate3D(ax, s=v, xyz=misc.tuple_from_col_vec(self.vertices[v]._pos), fontsize=10, xytext=(-2, 2),
                                textcoords='offset points', ha='right', va='bottom')
        
        for vertex in self.vertices.values():
            vertex.plot(axis=ax)

        return ax

    def get_directed_edge_list(self):
        edge_list = []
        for starting_vertex in self.edges.keys():
            for ending_vertex in self.edges[starting_vertex]:
                edge_list.append((starting_vertex, ending_vertex))
        return edge_list
    
    def get_undirected_edge_list(self):
        print("This way of generating the 'edge_list' might be slow...")
        edge_list = []
        for starting_vertex in self.edges.keys():
            for ending_vertex in self.edges[starting_vertex]:
                if (ending_vertex, starting_vertex) not in edge_list:
                    edge_list.append((starting_vertex, ending_vertex))
        return edge_list

    def show(self):
        plot.show()

