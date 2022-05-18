"""
Tue May 17th: Example of a GPS spoofing attack and mitigation
"""
import types

from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from functions import worldtime, mitigation_strategies, attacker, misc, graph, config, plot


config.PLOT_LIM = 2.5
config.OFFSET = [1.2, 0.7]
CONSTANT_ATTACK = True  # Uses simple constant bias attack instead of optimized (worst-case) attack


if __name__ == "__main__":
    name_list = misc.NameGenerator().generate(13)

    G = graph.Graph()
    positions = [(0.3, 0), (1.0, 0.7), (2.2, 0.5), (2.6, -0.6), (2.2, 1.5),  # A..E
                 (-0.7, 0.4), (-0.3, 2.1), (3.0, 0.7),  # ..H
                 (-0.1, 1.1), (0.6, 1.6), (1.5, -0.5),  # ..K
                 (0.7, 2.3), (1.5, 2.4)]  # ..M
    edges = [['A', 'F'], ['A', 'I'], ['A', 'B'], ['G', 'I'], ['G', 'J'], ['G', 'I'], ['J', 'I'], ['J', 'B'],
             ['B', 'C'], ['B', 'E'], ['B', 'K'], ['C', 'E'], ['C', 'H'], ['C', 'D'], ['C', 'K'], ['K', 'D'],
             ['I', 'F'], ['I', 'F'], ['D', 'H'], ['J', 'L'], ['L', 'M'], ['M', 'E']]

    for i in range(13):
        G.add_vertex(obj=mitigation_strategies.DroneWithRangeMitigation(name=name_list[i], position=positions[i],
                                                                        trajectory=misc.fixed_point_trajectory(point=positions[i]),
                                                                        perfect_init_conditions=True, process_noise=[1e-7, 1e-7, 1e-8, 1e-8],
                                                                        poles=[-3, -1.75, -1.5, -2], init_cov=[1e-4, 1e-4, 1e-7, 1e-7]))

    for e in edges:
        G.add_edge(e[0], e[1])

    attacker = attacker.Attacker()

    if CONSTANT_ATTACK:
        def design_constant_attack(self, _):
            _attack = []
            for _ in self.compromised_drones:
                _attack += [0.7, -0.7]
            self.attack_vector = misc.column(_attack)
            return
        attacker.design_attack = types.MethodType(design_constant_attack, attacker)

    G_m = deepcopy(G)
    attacker_m = deepcopy(attacker)
    attacker.add_compromised_drone(G.vertices['B'])
    attacker_m.add_compromised_drone(G_m.vertices['B'])

    for v in G.vertices.values():
        v.mitigation = False

    for v in G_m.vertices.values():
        v.mitigation = True

    plot_1 = G.draw()
    G_m.draw(axis=plot_1)

    def animate(_):
        for _ in range(5):
            worldtime.step()
            for vertex in G.vertices.values():
                vertex.update_physics()
            for vertex in G_m.vertices.values():
                vertex.update_physics()

        for vertex in G.vertices.values():
            attacker.design_attack(G.get_objects(attacker.compromised_drones))
            vertex.update_logic()
        for vertex in G_m.vertices.values():
            attacker_m.design_attack(G_m.get_objects(attacker_m.compromised_drones))
            vertex.update_logic()

        plot_1.cla()
        G.draw(axis=plot_1, alpha=0.2)
        G_m.draw(axis=plot_1)

    anim = animation.FuncAnimation(plt.gcf(), animate, tqdm(range(200)))
    anim.save("test.mp4", fps=30, dpi=200)
