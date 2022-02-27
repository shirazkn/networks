import networkx as nx
import matplotlib.pyplot as plt


def graph_undirected(graph):
    """
    The networkx package can plot aesthetically pleasing graphs
    but we want to plot frameworks (i.e. the vertices have positions)
    """
    nx_graph = nx.Graph(graph.edges)
    nx.draw(nx_graph, with_labels=True)
    plt.show()


def framework_undirected(framework):
    """Eg: for vertex in framework.vertices:
               ....
               for edge in framework.edges[vertex]:
                   ....
    """
    print("Note: currently ignoring node positions!")
    graph_undirected(framework)

    """
    Can networkx plot graphs + allow us to control the positions of the vertices? 
    Find out... https://networkx.org/documentation/latest/reference/index.html
    if not, do the following:
    - Plot a circle using matplotlib.pyplot, at framework.positions[framework.order(index())]
    - Add a small label (name of vertex) inside or outside the circle
    - Plot an edge from vertex to each of the vertices in framework.edges[vertex]
    """