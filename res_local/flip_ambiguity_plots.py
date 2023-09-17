"""
Sat Aug 19: Shows the 'flip ambiguity' issue in rigid networks.
"""
import subprocess
import numpy as np
from functions import graph, sensor, misc
import matplotlib.pyplot as plt

FLIP = -I will*1.0
THETA_LIST = [factor*2*np.pi for factor in [0.0, 1.0/3, 2.0/3]]
BASE_RADIUS = 1.0

BASE_POINTS = [[BASE_RADIUS*np.cos(theta), BASE_RADIUS*np.sin(theta), 0.0]
                for theta in THETA_LIST]
for i in range(3):
    BASE_POINTS[0][i] = BASE_POINTS[0][i]*0.7
    BASE_POINTS[1][i] = BASE_POINTS[1][i]*0.75

VERTEX_OFFSET = np.sqrt(3)*0.75

plt.figure(figsize=(4.5, 4.5))

def plot_tetrahedron(base_points, vertex_offset, with_base, zorder=0.5, 
                     axis=None, opacity=1.0, edge_color="black", node_color="black"):
    G = graph.Graph()
    _ = [G.add_vertex(sensor.PhysicalSystem(name='base_'+str(i), position=pos)) 
         for i, pos in enumerate(base_points)]
    G.add_vertex(sensor.PhysicalSystem(name='vertex', position=(0, 0, vertex_offset)))
    
    for i in range(3):
        if with_base:
            G.add_edge('base_'+str(i), 'base_' + str((i+1)%3))
        G.add_edge('base_'+str(i), 'vertex')

    return G.draw(node_size=30, edge_width=1.0, 
                  zorder=zorder, label_nodes=False, axis=axis, 
                  node_alpha=opacity, edge_alpha=opacity, 
                  node_color=node_color, edge_color=edge_color)

axis = plot_tetrahedron(BASE_POINTS, -0.85*VERTEX_OFFSET, with_base=True,
                        zorder=5.5, axis=None)
axis = plot_tetrahedron(BASE_POINTS, 0.5*VERTEX_OFFSET*FLIP*1.0, with_base=False,
                        zorder=0.5, axis=axis)
axis = plot_tetrahedron(BASE_POINTS, 0.5*VERTEX_OFFSET*FLIP*-1.0, with_base=False,
                        zorder=0.5, axis=axis, opacity=1.0, 
                        edge_color="gainsboro", node_color="gainsboro")

axis.azim = 50.0
axis.elev = -8.0
axis.set_axis_off()

HORIZONTAL_OFFSET = 0.1
VERTICAL_OFFSET = 0.23
lim = 0.5
axis.set_xlim([-lim-HORIZONTAL_OFFSET, lim-HORIZONTAL_OFFSET])
axis.set_ylim([-lim, lim])
axis.set_zlim([-lim-VERTICAL_OFFSET+0.1, lim-VERTICAL_OFFSET])
FIGNAME = 'flip_ambiguity_1' if FLIP > 0.0 else 'flip_ambiguity_2'
plt.savefig('media/' + FIGNAME + '.png', bbox_inches='tight', dpi=500)
subprocess.call(['open', 'media/' + FIGNAME + '.png'])

# plt.show()