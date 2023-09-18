import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from copy import deepcopy

from matplotlib import animation
from IPython import display  
# Just comment this if you don't have IPython, not sure if it still works either
from config import constants
from functions import misc


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

    if constants.MARKER_TYPE == 'drone':
        kwargs["with_labels"] = False
        kwargs["node_size"] = 300
        kwargs["node_color"] = '#ffffff'
        kwargs["width"] = 0.9
        kwargs["style"] = 'dashed'
        kwargs["node_shape"] = 's'
        kwargs["edge_color"] = '#2f89c5'

    elif constants.MARKER_TYPE == 'ieee_labeled_graph':
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

    if constants.MARKER_TYPE == 'drone':
        imgbox = get_image_box("media/drone.png", zoom=0.12)
        for v in graph.vertices.values():
            plot_img((v._pos[0][0], v._pos[1][0]), imgbox, axis, zorder=30)

    for vertex in graph.vertices.values():
        vertex.plot(axis=axis)

    axis.set_xlim((-constants.PLOT_LIM + constants.OFFSET[0], constants.PLOT_LIM + constants.OFFSET[0]))
    axis.set_ylim((-constants.PLOT_LIM + constants.OFFSET[1], constants.PLOT_LIM + constants.OFFSET[1]))
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


def plot_point3D(point, axis, **style):
    axis.scatter([point[0]], [point[1]], [point[2]], **style)


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


from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s
    Code from https://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot
    '''

    def __init__(self, s, xyz, my_axis=None, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        
        self.my_axis = my_axis

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        # import pdb; pdb.set_trace()
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.my_axis.get_proj())
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)


def annotate3D(ax, s, *args, **kwargs):
    '''Adds anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, my_axis=ax, *args, **kwargs)
    ax.add_artist(tag)


def border3D(axis):
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()
    zlim = axis.get_zlim()

    cube_corners = [
        [xlim[0], ylim[0], zlim[0]],
        [xlim[0], ylim[0], zlim[1]],
        [xlim[0], ylim[1], zlim[0]],
        [xlim[0], ylim[1], zlim[1]],
        [xlim[1], ylim[0], zlim[0]],
        [xlim[1], ylim[0], zlim[1]],
        [xlim[1], ylim[1], zlim[0]],
        [xlim[1], ylim[1], zlim[1]]
    ]

    cube_edges = [
        [cube_corners[0], cube_corners[1]],
        [cube_corners[0], cube_corners[2]],
        [cube_corners[0], cube_corners[4]],
        [cube_corners[1], cube_corners[3]],
        [cube_corners[1], cube_corners[5]],
        [cube_corners[2], cube_corners[3]],
        [cube_corners[2], cube_corners[6]],
        [cube_corners[3], cube_corners[7]],
        [cube_corners[4], cube_corners[5]],
        [cube_corners[4], cube_corners[6]],
        [cube_corners[5], cube_corners[7]],
        [cube_corners[6], cube_corners[7]]
    ]

    for edge in cube_edges:
        axis.plot(*zip(*edge), color='black')

    axis.set_xlim3d(xlim)
    axis.set_ylim3d(ylim)
    axis.set_zlim3d(zlim)
    return axis