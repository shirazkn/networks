from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import numpy as np
from tqdm import tqdm
from copy import deepcopy
from functions import graph, misc, plot, sensor


def check_l2l1_condition(G):
    def _3D_rotation_about_axis(theta, axis):
        _2D_rotation = misc.rotation_matrix(theta)
        _3D_rotation = np.zeros([3, 3])
        for i in [0, 1]:
            for j in [0, 1]:
                _3D_rotation[(axis+i+1)%3][(axis+j+1)%3] = _2D_rotation[i][j]
        
        return _3D_rotation

    def _get_random_3D_rotations(num_monte_carlo):
        rand_angles = [np.random.random(num_monte_carlo)*np.pi*2.0 for _ in range(3)]
        rand_rotations = []
        for i in range(num_monte_carlo):
            rand_rotations.append(np.identity(3))
            for j in [0, 1, 2]:
                rand_rotations[-1] @= _3D_rotation_about_axis(rand_angles[j][i], j)
        return rand_rotations             

    print("Assuming the sensor network is contained within an 8x8x8 cube...")
    offset = misc.column([4, 4, 4])
    s_max = len(G.vertices)
    resolution = 50

    for R in tqdm(_get_random_3D_rotations(500)):
        positions = [R @ (v._pos[:3]-offset) for v in G.vertices.values()]
        
        for c_x in np.linspace(-4, 4, resolution):
            for c_y in np.linspace(-4, 4, resolution):
                distances = []
                for p in positions:
                    distances.append(np.linalg.norm(p[:2] - misc.column([c_x, c_y])))
                
                distances.sort()
                s = 0
                while True:
                    if sum(distances[:s+1]) < sum(distances[s+1:]):
                        s += 1
                    else: 
                        break

                if s < s_max:
                    s_max = s
    print(f"The maximal value of s is {s_max}!")

def update_estimates(G, error_vector):
                for name in G.vertices:
                    index = G.index(name)
                    G.vertices[name].ekf.x[0:3] += misc.column(error_vector[3*index: 3*(index+1)])


class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

            return np.min(zs)
        
def draw_shifted_graph(G1, G2):
    # axis = G1.draw(node_color="black", node_size=10, zorder=1)
    # G2_wo_edges = graph.Graph()

    # for name in G2.vertices:
    #     G2_wo_edges.add_vertex(obj=sensor.Drone2D(position=deepcopy(
    #     misc.tuple_from_col_vec(G2.vertices[name]._pos[0:3])
    #     ), name=name))

    # G2_wo_edges.draw(axis=axis, node_size=55, node_color="orange", 
    #                 node_alpha=0.9, zorder=0.5, label_nodes=False)
    
    # plot.plt.tight_layout()
    # axis.set_xlim([0.0001, 6.999])
    # axis.set_ylim([0.0001, 6.999])
    # axis.set_zlim([0.0001, 6.999])
    # axis.text(*(-0.46, -0.45, -0.51), '0')
    # axis.text(*(-0.51, 7.485, -0.55), '0')

    # axis.azim = -108.0
    # axis.elev = 15.0

    # axis.grid(True)
    # for _a in [axis.xaxis, axis.yaxis, axis.zaxis]:
    #     _a._axinfo["grid"].update({"linewidth": 0.2, "alpha": 0.8})

    # label_padding = [-5, -2]
    # axis.set_xlabel(r"$x$", labelpad=label_padding[0])
    # axis.set_ylabel(r"$y$", labelpad=label_padding[0])
    # axis.set_zlabel(r"$z$", labelpad=label_padding[0])
    # axis.tick_params(axis="x", pad=label_padding[1])
    # axis.tick_params(axis="y", pad=label_padding[1])
    # axis.tick_params(axis="z", pad=label_padding[1])

    # for name in G1.vertices:
    #     tail_point = G2.vertices[name]._pos
    #     head_point = G1.vertices[name].ekf.x[0:3]
    #     if np.linalg.norm(head_point - tail_point) > 0.5:
    #         arrow = Arrow3D([tail_point[0][0], head_point[0][0]], 
    #                         [tail_point[1][0], head_point[1][0]], 
    #                         [tail_point[2][0], head_point[2][0]], 
    #                         mutation_scale=10, 
    #                         lw=1, arrowstyle="-|>", color="r")
    #         axis.add_artist(arrow)

    # return axis
    raise NotImplementedError  # commented code is from the centralized version...

def draw_shifted_graph_ADMM(true_graph, estimates_init, estimates):
    axis = true_graph.draw(node_color="black", node_size=8, zorder=1)
    estimates_init.draw(axis=axis, node_size=40, node_color="orange", 
                    node_alpha=0.8, zorder=0.5, label_nodes=False)
    
    plot.plt.tight_layout()
    axis.set_xlim([0.0001, 6.3])
    axis.set_ylim([0.0001, 6.3])
    axis.set_zlim([0.0001, 6.3])
    axis.text(*(-0.47, -0.45, -0.53), '0')
    axis.text(*(-0.38, 7.085, -0.52), '0')

    axis.azim = -108.0
    axis.elev = 15.0

    axis.grid(True)
    for _a in [axis.xaxis, axis.yaxis, axis.zaxis]:
        _a._axinfo["grid"].update({"linewidth": 0.2, "alpha": 0.8})

    label_padding = [-5, -2]
    axis.set_xlabel(r"$x$", labelpad=label_padding[0])
    axis.set_ylabel(r"$y$", labelpad=label_padding[0])
    axis.set_zlabel(r"$z$", labelpad=label_padding[0])
    axis.tick_params(axis="x", pad=label_padding[1])
    axis.tick_params(axis="y", pad=label_padding[1])
    axis.tick_params(axis="z", pad=label_padding[1])

    for vtx_name in true_graph.vertices:
        tail_point = estimates_init.vertices[vtx_name]._pos
        head_point = estimates.vertices[vtx_name]._pos
        if np.linalg.norm(head_point - tail_point) > 0.2:
            arrow = Arrow3D([tail_point[0][0], head_point[0][0]], 
                            [tail_point[1][0], head_point[1][0]], 
                            [tail_point[2][0], head_point[2][0]], 
                            mutation_scale=10, 
                            lw=0.9, arrowstyle="-|>", color="r")
            axis.add_artist(arrow)

    return axis

def draw_only_graph(g):
    axis = g.draw(node_color="black", node_size=20, zorder=1, label_nodes=False)
    
    plot.plt.tight_layout()
    axis.set_xlim([0.0001, 6.999])
    axis.set_ylim([0.0001, 6.999])
    axis.set_zlim([0.0001, 6.999])
    axis.azim = -108.0
    axis.elev = 15.0
    axis.set_axis_off()

    return axis

def get_nonzero_blocks(x_k, tol=0.5):
    length = int(len(x_k)/3)
    nonzero_blocks = []
    for i in range(length):
        if np.linalg.norm(x_k[3*i: 3*(i+1)]) > tol:
            nonzero_blocks.append(i)

    return nonzero_blocks

def get_graph(positions, edges):
    name_list = misc.NameGenerator(name_type='number').generate(len(positions))
    G = graph.Graph()
    for i in range(len(positions)):
        G.add_vertex(obj=sensor.Drone2D(position=positions[i], name=name_list[i]))

    for e in edges:
        G.add_edge(e[0], e[1])
    
    return G

def get_distance_rigidity_matrix(G, using_estimates=False, verbose=False):
    nx_G = graph.nxGraph(G.edges)
    edge_list = list(nx_G.edges)
    dimensions = list(G.vertices.values())[0].dimensions
    
    R = np.zeros([len(edge_list), dimensions*len(G.vertices)])
    for i, e in enumerate(edge_list):
        if using_estimates:
            edge_pos = G.vertices[e[0]].ekf.x[0:dimensions].T - G.vertices[e[1]].ekf.x[0:dimensions].T
        else:
            edge_pos = G.vertices[e[0]]._pos.T - G.vertices[e[1]]._pos.T

        R[i, (dimensions * G.index(e[0]) ):(dimensions * (G.index(e[0])+1) )] = edge_pos
        R[i, (dimensions * G.index(e[1]) ):(dimensions * (G.index(e[1])+1) )] = -1*edge_pos
    
    if verbose:
        if dimensions == 3:
            print(f"Graph has {len(G.vertices)} vertices and {len(edge_list)} edges.", 
                  "First 7 eigenvalues of R'*R are: \n", np.linalg.eigvalsh(R.T @ R)[:7]) 
        else:
            print(f"Graph has {len(G.vertices)} vertices and {len(edge_list)} edges.",
                  "First 4 eigenvalues of R'*R are: \n", np.linalg.eigvalsh(R.T @ R)[:4])
        
    return R

def get_incidence_matrix(G):
    nx_G = graph.nxGraph(G.edges)
    edge_list = list(nx_G.edges)
    I = np.zeros([len(G.vertices), len(edge_list)])
    for i, e in enumerate(edge_list):
        I[G.index(e[0]), i] = 1
        I[G.index(e[1]), i] = -1
    return I

def get_phi_D(G, using_estimates):
    nx_G = graph.nxGraph(G.edges)
    edge_list = list(nx_G.edges)
    phi = []
    for e in edge_list:
        edge_vec = G.vertices[e[0]]._pos - G.vertices[e[1]]._pos
        if using_estimates:
            edge_vec = G.vertices[e[0]].ekf.x[0:3] - G.vertices[e[1]].ekf.x[0:3]
        phi.append(float((edge_vec.T @ edge_vec))*0.5)
    return np.array(phi)

def get_stacked_position_vector(G):
    pos = np.zeros([2*len(G.vertices), 1])
    for v in G.vertices:
        pos[2*G.index(v):(2*G.index(v)+2)] = deepcopy(G.vertices[v]._pos)
    return pos

def plot_red(self, axis, **kwargs):
    if self.gps_timer.get_elapsed_time() < 0.35:
        plot.plot_point3D(misc.tuple_from_col_vec(
            self._pos + misc.column([0, 0, 0])
            ), axis=axis, color=(0.95, 0.1, 0.2), s=15, edgecolor=(0.85, 0.05, 0.15))
    
def get_mean_var(vec):
    mean = np.mean(vec)
    sum_of_squares = 0.0
    for elem in vec:
        sum_of_squares += (elem - mean)**2

    return mean, (sum_of_squares/(len(vec) - 1))

