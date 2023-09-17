"""
Mon, Apr 17 2023: Same as l2l1_rigidity, using ADMM to determine sparse localization errors in a distributed manner
-------------------------------------------------------------------------------------------------------------------
"""

from functions import graph, plot, sensor, misc
import numpy as np
import networkx as nx

from matplotlib import animation
from tqdm import tqdm
import subprocess
import types
from functions import worldtime

positions = [
    (1.4, 4, 7), #1
    (4.4, 7, 8.6), #2
    (3.75, 6.4, 5.7), #3
    (2.75, 2, 3), #4
    (4.6, 4.45, 6.7), # 5
    (5.0, 5.85, 3.6),
    (5.05, 3.4, 4.2), # 7
    (6.8, 6.6, 6.75),
    (6.9, 5.4, 1.4), # 9
    (7.2, 2.4, 4.75), # 10
    (8.1, 4.9, 7.8),
    (6.65, 5.4, 5.2), # 12
    (5.75, 7.25, 5.0) # 13
]
PLOT_LIM= 10.0

name_list = misc.NameGenerator(name_type='number').generate(len(positions))

edges = [['1', '2'], ['1','3'], ['1', '4'], ['1', '5'], ['1', '6'], ['1','7'],
        ['2', '3'], ['2','5'], ['2', '8'], ['2', '11'], ['2', '12'],
        ['3', '5'], ['3', '6'], ['3', '8'], ['4','5'], ['4','6'], ['4', '9'], ['5','6'], ['5','7'], ['5', '8'], ['5', '11'],
        ['6', '7'], ['6', '9'], ['6', '10'], ['7', '8'], ['7','10'], ['8', '11'],
        ['9', '10'], ['9', '12'], ['10','12'], ['10','11'], ['11', '12'],
        ['13', '6'], ['13', '8'], ['13', '12'], ['13', '3']
        ]

print(f"Graph has {len(positions)} vertices and {len(edges)} edges. (Running ADMM-based algorithm...)")

# Make the graphs
G = graph.Graph()
for i in range(len(name_list)):
    G.add_vertex(obj=sensor.Drone2D(position=positions[i], name=name_list[i]))

# Add edges
for e in edges:
    G.add_edge(e[0], e[1])


def random_vector_in_box(a):
    vec = 2*a*np.random.default_rng().random([3,1])
    return vec - misc.column([a, a, a])


# G.pos is True, G.ekf.x is False, and G_est.pos is False
FAULTY_DRONES = ['12', '13', '8', '11']
BIAS_VECTORS = {name: random_vector_in_box(1.0) for name in FAULTY_DRONES} 
for name in G.vertices:
    G.vertices[name].ekf.x = np.concatenate([G.vertices[name]._pos, np.zeros([3, 1])])
    if name in FAULTY_DRONES:
        G.vertices[name].ekf.x += np.concatenate([BIAS_VECTORS[name], np.zeros([3, 1])])


# --------- Initial (biased) estimates are stored in another graph

from copy import deepcopy
G_est = graph.Graph()
for name in G.vertices:
    G_est.add_vertex(obj=sensor.Drone2D(position=deepcopy(
        misc.tuple_from_col_vec(G.vertices[name].ekf.x[0:3])
        ), name=name))


# ----------------------------------------------------------------------------------------------------------------------
# --------- Setting up the optimization problem: -----------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

import cvxpy as cp

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# ------------------- Update variables in loop... ------
def update_primal_x(v):
    v.state.x = misc.column(v.cvx.x.value)

def update_primal_w(v):
    for nbr in v.neighbors:
        v.state.w[nbr.name] = misc.column(v.cvx.w[nbr.name].value)

def update_estimates(_G):
    for name in _G.vertices:
        _G.vertices[name].ekf.x[0:3] += misc.column(_G.vertices[name].state.x)

def get_residual(G, i, j):
    true_edge = G.vertices[i]._pos - G.vertices[j]._pos
    est_edge = G.vertices[i].ekf.x[0:3] - G.vertices[j].ekf.x[0:3]
    return (float(true_edge.T @ true_edge) - float(est_edge.T @ est_edge))*0.5

# ------------------- Initialization -------------------
for v in G.vertices.values():
    v.cvx = Namespace(x = cp.Variable((3,1)), 
                      w = {nbr.name: cp.Variable((3,1)) for nbr in v.neighbors})
    v.state = Namespace(x = np.zeros([3, 1]), 
                        w = {nbr.name: np.zeros([3, 1]) for nbr in v.neighbors}, 
                        mu = {nbr.name : np.zeros([3,1]) for nbr in v.neighbors}, 
                        lam = {nbr.name : 0.0 for nbr in v.neighbors})

# Main Loop --------------------------------------------
# tol = 0.001
rho_1 = 1.0
rho_2 = rho_1
from tqdm import tqdm

for _ in tqdm(range(25)):
    for _ in range(4):
        # -- Update primal_1 --------------------------------------
        for vtx_name in G.vertices:
            vtx = G.vertices[vtx_name]
            objective = cp.norm(vtx.cvx.x)
            for nbr in vtx.neighbors:
                rig_edge = vtx.ekf.x[0:3] + vtx.state.x - (nbr.ekf.x[0:3] + nbr.state.x)
                rig_norm = np.linalg.norm(rig_edge)
                residual = np.linalg.norm(vtx._pos - nbr._pos) - np.linalg.norm(vtx.ekf.x[0:3] - nbr.ekf.x[0:3])
                distance_constraint = (rig_edge/rig_norm).T @ (vtx.cvx.x - vtx.state.w[nbr.name]) - residual

                objective += (
                    (rho_1/2.0)*cp.power(cp.norm(nbr.state.w[vtx_name] - vtx.cvx.x), 2)
                    + vtx.state.mu[nbr.name].T @ (nbr.state.w[vtx_name] - vtx.cvx.x)
                    + (rho_2/2.0)*cp.power(cp.norm(distance_constraint),2)
                    + vtx.state.lam[nbr.name] * distance_constraint
                            )
            
            cp.Problem(cp.Minimize(objective), []).solve()
        
        for vtx_name in G.vertices: 
            update_primal_x(G.vertices[vtx_name])

        # -- Update primal_2 --------------------------------------
        for vtx_name in G.vertices:
            vtx = G.vertices[vtx_name]
            objective = 0.0
            for nbr in vtx.neighbors:
                rig_edge = vtx.ekf.x[0:3] + vtx.state.x - (nbr.ekf.x[0:3] + nbr.state.x)
                rig_norm = np.linalg.norm(rig_edge)
                residual = np.linalg.norm(vtx._pos - nbr._pos) - np.linalg.norm(vtx.ekf.x[0:3] - nbr.ekf.x[0:3])
                distance_constraint = (rig_edge/rig_norm).T @ (vtx.state.x - vtx.cvx.w[nbr.name]) - residual
                
                objective += (
                    (rho_1/2.0)*cp.power(cp.norm(vtx.cvx.w[nbr.name] - nbr.state.x), 2)
                    + nbr.state.mu[vtx.name].T @ (vtx.cvx.w[nbr.name] - nbr.state.x)
                    + (rho_2/2.0)*cp.power(cp.norm(distance_constraint), 2)
                    + vtx.state.lam[nbr.name] * distance_constraint
                )
                
                # constraints.append(R_block.T @ (vtx.state.x - vtx.cvx.w[nbr.name]) - get_residual(G, vtx_name, nbr.name) == 0.0)
                # constraints.append(R_block.T @ (vtx.state.x - vtx.cvx.w[nbr.name]) - get_residual(G, vtx_name, nbr.name) >= -tol)
            
            cp.Problem(cp.Minimize(objective), []).solve()
        
        for vtx_name in G.vertices:
            update_primal_w(G.vertices[vtx_name])

        # -- Update multipliers -----------------------------------
        for vtx_name in G.vertices:
            vtx = G.vertices[vtx_name]
            for nbr in vtx.neighbors:   
                rig_edge = vtx.ekf.x[0:3] + vtx.state.x - (nbr.ekf.x[0:3] + nbr.state.x)
                rig_norm = np.linalg.norm(rig_edge)
                residual = np.linalg.norm(vtx._pos - nbr._pos) - np.linalg.norm(vtx.ekf.x[0:3] - nbr.ekf.x[0:3])
                distance_constraint = (rig_edge/rig_norm).T @ (vtx.state.x - vtx.state.w[nbr.name]) - residual

                vtx.state.mu[nbr.name] += rho_1*(nbr.state.w[vtx_name] - vtx.state.x)
                vtx.state.lam[nbr.name] += rho_2*(float(distance_constraint))

    update_estimates(G)


# ---------------------------------------------------------------
# ------- Plotting arrows: --------------------------------------

# update_estimates(G)

axis = G.draw()
G_est.draw(axis=axis)
axis.set_xlim([0, PLOT_LIM])
axis.set_ylim([0, PLOT_LIM])
axis.set_zlim([0, PLOT_LIM])


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

for name in G.vertices:
    tail_point = G_est.vertices[name]._pos
    head_point = G.vertices[name].ekf.x[0:3]
    if np.linalg.norm(head_point - tail_point) > 0.1:
        arrow = Arrow3D([tail_point[0][0], head_point[0][0]], 
                        [tail_point[1][0], head_point[1][0]], 
                        [tail_point[2][0], head_point[2][0]], 
                        mutation_scale=10, 
                        lw=1, arrowstyle="-|>", color="r")
        axis.add_artist(arrow)
plot.show()
