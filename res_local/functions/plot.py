import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from copy import deepcopy

from matplotlib import animation
from IPython import display
from functions import misc, config


plt.rcParams.update({"text.usetex": True})


def new_figure(number=None):
    return plt.figure(number)


def draw_graph(graph, axis, title, plot_estimates=False, **kwargs):
    """
    Plots graph with vertices placed in a specific configuration
    Potentially supports 3D configurations(?)
    """
    nx_graph = nx.Graph(graph.edges)

    # If the vertices have 'positions' use those
    positions = None
    if graph.vertices[graph.order[0]]._pos is not None:
        positions = {}
        for vertex in graph.vertices.keys():
            positions[vertex] = misc.tuple_from_col_vec(graph.vertices[vertex]._pos)

    kwargs["with_labels"] = True

    if config.MARKER_TYPE == 'drone':
        kwargs["with_labels"] = False
        kwargs["node_size"] = 300
        kwargs["node_color"] = '#ffffff'
        kwargs["width"] = 0.9
        kwargs["style"] = 'dashed'
        kwargs["node_shape"] = 's'
        kwargs["edge_color"] = '#2f89c5'

    elif config.MARKER_TYPE == 'ieee_labeled_graph':
        kwargs["node_size"] = 300
        kwargs["node_color"] = kwargs.get("node_color", '#000000')
        kwargs["edge_color"] = "tab:gray"
        kwargs["width"] = 1.0
        kwargs['style'] = 'solid'
        kwargs['font_color'] = "white"
        kwargs['font_size'] = 15
        kwargs['font_weight'] = 'bold'

    else:
        kwargs["node_size"] = 150
        kwargs["node_color"] = kwargs.get("node_color", '#000000')
        kwargs["edge_color"] = "tab:gray"
        kwargs["width"] = 1.0
        kwargs['style'] = 'solid'
        kwargs['font_color'] = "white"
        kwargs['font_size'] = 10
        kwargs['font_weight'] = 'bold'

    nx.draw(nx_graph, positions, ax=axis, **kwargs)
    fig = plt.gcf()
    if not axis:
        axis = fig.gca()

    if config.MARKER_TYPE == 'drone':
        imgbox = get_image_box("media/drone.png", zoom=0.12)
        for v in graph.vertices.values():
            plot_img((v._pos[0][0], v._pos[1][0]), imgbox, axis, zorder=30)

    for vertex in graph.vertices.values():
        vertex.plot(axis=axis)

    axis.set_xlim((-config.PLOT_LIM + config.OFFSET[0], config.PLOT_LIM + config.OFFSET[0]))
    axis.set_ylim((-config.PLOT_LIM + config.OFFSET[1], config.PLOT_LIM + config.OFFSET[1]))
    plt.xticks()
    plt.yticks()
    plt.grid()


def show():
    plt.show()


def get_axis():
    return plt.gca()


def get_image_box(path, zoom):
    img = plt.imread(path, format='png')
    return OffsetImage(img, zoom=zoom)


def plot_point(point, **style):
    plt.scatter([point[0]], [point[1]], zorder=10, **style)


def plot_img(point, imgbox, axis, **style):
    imgbox.image.axes = axis
    ab = AnnotationBbox(imgbox, point,
                        xybox=(0., 0.),
                        xycoords='data',
                        boxcoords="offset points",
                        frameon=False,
                        pad=5.0,
                        **style)
    axis.add_artist(ab)


def plot_line(points, style):
    plt.plot([point[0] for point in points], [point[1] for point in points], style, zorder=-10, linewidth=0.5)


def display_animation(anim):
    # Currently unused...
    # Note : interval=1000.0/30
    FFMpegWriter = animation.writers['ffmpeg']
    FFMpegWriter = FFMpegWriter(fps=30)
    video = anim.to_html5_video()
    display.display(display.HTML(video))
