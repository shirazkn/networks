"""
Tue May 17th: Example of a GPS spoofing attack and mitigation
"""
import types

from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from functions import worldtime, mitigation_strategies, attacker, misc, graph, config, plot
import numpy.linalg as la

ATTACK_START_TIME = 100
TIMESTEPS = 600

config.PLOT_LIM = 2.3
config.OFFSET = [1.2, 0.9]
config.MARKER_TYPE = 'drone'

CONSTANT_ATTACK = True  # Uses simple constant bias attack instead of optimized (worst-case) attack
# If false, uses cvxpy to optimize attack that minimizes the range-residuals
PLOT_UNMITIGATED_NETWORK = False
SECURED_NETWORK = True


if __name__ == "__main__":
    name_list = misc.NameGenerator().generate(15)

    G = graph.Graph()
    positions = [(0.3, 0), (1.0, 0.7), (1.8, 0.3), (2.6, -0.6), (1.5, 1.3),  # A..E
                 (-0.7, 0.4), (-0.3, 2.1), (3.0, 0.7),  # ..H
                 (-0.1, 1.1), (0.5, 1.5), (1.5, -0.5),  # ..K
                 (0.5, 2.5), (1.4, 2.1), (2.1, 1.32), (2.7, 1.9),

                 ]  # ..N
    edges = [['A', 'F'], ['A', 'I'], ['A', 'B'], ['G', 'I'], ['G', 'J'], ['G', 'I'], ['J', 'I'], ['J', 'B'],
             ['B', 'C'], ['B', 'E'], ['B', 'K'], ['C', 'E'], ['C', 'H'], ['C', 'D'], ['C', 'K'], ['K', 'D'],
             ['I', 'F'], ['I', 'F'], ['D', 'H'], ['J', 'L'], ['L', 'M'], ['M', 'E'], ['N', 'E'], ['N', 'H'],
             ['N', 'O'], ['O', 'H']]

    for i in range(len(name_list)):
        G.add_vertex(obj=mitigation_strategies.DroneWithRangeMitigation(name=name_list[i], position=positions[i],
                                                                        trajectory=misc.fixed_point_trajectory(point=positions[i]),
                                                                        perfect_init_conditions=False, process_noise=[1e-8, 1e-8, 1e-8, 1e-8],
                                                                        poles=[-2, -3.75, -5.5, -6], init_cov=[1e-4, 1e-4, 1e-7, 1e-7]))

    for e in edges:
        G.add_edge(e[0], e[1])

    attacker = attacker.Attacker()

    if CONSTANT_ATTACK:
        def design_constant_attack(self, timestep):
            ATTACK_VECTORS = {"L": misc.column([-0.20, 0.03]), "M": misc.column([-0.20, 0.03])}
            ATTACK_VECTORS_2 = {"L": misc.column([-0.12, 0.10]), "M": misc.column([-0.12, 0.10])}
            self.attack_vectors = []
            for drone in self.compromised_drones:
                if timestep < 400:
                    self.attack_vectors.append(ATTACK_VECTORS[drone])
                else:
                    self.attack_vectors.append(ATTACK_VECTORS_2[drone])
            return
        attacker.design_attack = types.MethodType(design_constant_attack, attacker)

    G_m = deepcopy(G)
    attacker_m = deepcopy(attacker)

    for v in G.vertices.values():
        v.if_mitigation = False

    for v in G_m.vertices.values():
        v.if_mitigation = True

    if SECURED_NETWORK:
        G.add_edge('L', 'E')
        G_m.add_edge('L', 'E')
        G.add_edge('J', 'M')
        G_m.add_edge('J', 'M')

    plot_1 = G.draw()
    G_m.draw(axis=plot_1)

    def animate(timestep):
        if timestep == ATTACK_START_TIME:
            attacker.add_compromised_drone(G.vertices['L'])
            attacker.add_compromised_drone(G.vertices['M'])
            attacker_m.add_compromised_drone(G_m.vertices['L'])
            attacker_m.add_compromised_drone(G_m.vertices['M'])

        for _ in range(5):
            worldtime.step()
            if PLOT_UNMITIGATED_NETWORK:
                for vertex in G.vertices.values():
                    vertex.update_physics()
            for vertex in G_m.vertices.values():
                vertex.update_physics()

        if PLOT_UNMITIGATED_NETWORK:
            for vertex in G.vertices.values():
                attacker.design_attack(timestep)
                vertex.update_logic()
        for vertex in G_m.vertices.values():
            attacker_m.design_attack(timestep)
            vertex.update_logic()

        plot_1.cla()
        if PLOT_UNMITIGATED_NETWORK:
            G.draw(axis=plot_1, alpha=0.2)
        G_m.draw(axis=plot_1)

    anim = animation.FuncAnimation(plt.gcf(), animate, tqdm(range(TIMESTEPS)))
    anim.save("test.mp4", fps=30, dpi=200)

    import subprocess
    subprocess.call(["open", "test.mp4"])

