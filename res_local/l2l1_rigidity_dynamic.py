"""
Fri, Apr 7 2023: Same as l2l1_rigidity, but where the agents are moving
--------------------------------------------------------------------------
The goal is to use an iterative optimization algorithm to identify localization errors
"""

from functions import graph, plot, sensor, misc
import numpy as np
import networkx as nx

from matplotlib import animation
from tqdm import tqdm
import subprocess
import types
from functions import worldtime

def get_distance_rigidity_matrix(G):
    nx_G = nx.Graph(G.edges)
    edge_list = list(nx_G.edges)
    dimensions = list(G.vertices.values())[0].dimensions
    R = np.zeros([len(edge_list), dimensions*len(G.vertices)])
    for i, e in enumerate(edge_list):
        edge_pos = G.vertices[e[0]]._pos.T - G.vertices[e[1]]._pos.T
        R[i, (dimensions * G.index(e[0]) ):(dimensions * (G.index(e[0])+1) )] = edge_pos
        R[i, (dimensions * G.index(e[1]) ):(dimensions * (G.index(e[1])+1) )] = -1*edge_pos
    return R


def get_incidence_matrix(G):
    nx_G = nx.Graph(G.edges)
    edge_list = list(nx_G.edges)
    I = np.zeros([len(G.vertices), len(edge_list)])
    for i, e in enumerate(edge_list):
        I[G.index(e[0]), i] = 1
        I[G.index(e[1]), i] = -1
    return I


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

print(f"Graph has {len(positions)} vertices and {len(edges)} edges.")

# Make the graphs
graph = graph.Graph()
for i in range(len(name_list)):
    PROCESS_NOISE = [0.01 for _ in range(3)] + [0.005 for _ in range(3)]
    graph.add_vertex(obj=sensor.Drone2D(position=positions[i], name=name_list[i],
                                    perfect_init_conditions=False, process_noise=PROCESS_NOISE, ins_var=0.001, gps_var=0.01, 
                                    init_cov=[0.1 for _ in range(6)]))

# Add edges
for e in edges:
    graph.add_edge(e[0], e[1])

FAULTY_DRONES = ['12', '13', '8', '11']
BIAS_VECTORS = {name: misc.column([0.20, -0.09, 0]) for name in FAULTY_DRONES}
def get_fake_gps(obj: sensor.Drone2D):
        return obj.ekf.x[0:obj.dimensions] + misc.random_gaussian(obj.gps_cov) + BIAS_VECTORS[obj.name]    

R = get_distance_rigidity_matrix(graph)
In = get_incidence_matrix(graph)

print("First 7 eigenvalues of R'*R are ", np.linalg.eigvalsh(R.T @ R)[:7])

VIDEO = True
if VIDEO:
    a = graph.draw()
    def animate(i):
        if i == 75:
            # self.compromised_drones.append(drone.name)
            # Whenever drone calls get_gps, call get_fake_gps instead:
            for drone in [graph.vertices[name] for name in FAULTY_DRONES]:
                drone.get_gps_measurement = types.MethodType(get_fake_gps, drone)
                drone.plot = types.MethodType(plot_red, drone)

        for _ in range(1):
            for _ in range(1):
                worldtime.step()
                for vertex in graph.vertices.values():
                    vertex.update_physics()

            for vertex in graph.vertices.values():
                vertex.update_logic()
        a.cla()
        graph.draw(axis=a)
        a.set_xlim([0, PLOT_LIM])
        a.set_ylim([0, PLOT_LIM])
        a.set_zlim([0, PLOT_LIM])

    anim = animation.FuncAnimation(plot.plt.gcf(), animate, tqdm(range(225)))
    anim.save("test.mp4", fps=30, dpi=200)
    subprocess.call(["open", "test.mp4"])

else:
    axis = graph.draw()
    axis.set_xlim([0, PLOT_LIM])
    axis.set_ylim([0, PLOT_LIM])
    axis.set_zlim([0, PLOT_LIM])
    plot.show()

