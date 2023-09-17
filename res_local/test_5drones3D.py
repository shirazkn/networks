"""
Tue, Apr 4 2023: Trying to see how NetworkX fares at plotting 3D frameworks
Answer: It works just fine!
"""
from config import constants
from functions import graph, sensor, misc
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import subprocess

from functions import worldtime

constants.PLOT_TRAJECTORIES = True


if __name__ == "__main__":
    name_list = misc.NameGenerator().generate(5)

    PROCESS_NOISE = [0.01 for _ in range(3)] + [0.005 for _ in range(3)]
    INS_VAR = 0.001
    GPS_VAR = 0.01
    constants.LIMIT_VEL_ACC = True
    constants.SPEED_LIMIT = 1.0
    constants.ACC_LIMIT = 0.5

    G = graph.Graph()
    # Moving drones:
    G.add_vertex(obj=sensor.Drone2D(position=(-5, 0, 10), name=name_list[0],
                                    trajectory=misc.lemniscate_of_bernoulli(a=7, offset=(-5, 0, 10), scale_y=1.2),
                                    perfect_init_conditions=True, process_noise=PROCESS_NOISE, 
                                    ins_var=INS_VAR, gps_var=GPS_VAR))
    G.add_vertex(obj=sensor.Drone2D(position=(5, 0, 10), name=name_list[1],
                                    trajectory=misc.lemniscate_of_bernoulli(a=-7, offset=(5, 0, 10), scale_y=1.2, time_offset=0.5), 
                                    perfect_init_conditions=True, process_noise=PROCESS_NOISE, 
                                    ins_var=INS_VAR, gps_var=GPS_VAR))

    # Stationary drones:
    G.add_vertex(obj=sensor.Drone2D(position=(-8, 0, 10), name=name_list[2],
                                    trajectory=misc.fixed_point_trajectory(point=(-8, 0, 10)),
                                    perfect_init_conditions=True, process_noise=PROCESS_NOISE, ins_var=INS_VAR, gps_var=GPS_VAR))
    G.add_vertex(obj=sensor.Drone2D(position=(8, 0, 10), name=name_list[3],
                                    trajectory=misc.fixed_point_trajectory(point=(8, 0, 10)),
                                    perfect_init_conditions=True, process_noise=PROCESS_NOISE, ins_var=INS_VAR, gps_var=GPS_VAR))
    G.add_vertex(obj=sensor.Drone2D(position=(0, -8, 10), name=name_list[4],
                                    trajectory=misc.fixed_point_trajectory(point=(0, -8, 10)),
                                    perfect_init_conditions=True, process_noise=PROCESS_NOISE, ins_var=INS_VAR, gps_var=GPS_VAR))

    G.add_edge("A", "C")
    G.add_edge("B", "D")
    G.add_edge("C", "D")
    G.add_edge("C", "E")
    G.add_edge("D", "E")

    a = G.draw()
    # Not bothering to animate 3D right now...
    # plt.show()
    def animate(i):
        for _ in range(2):
            for _ in range(1):
                worldtime.step()
                for vertex in G.vertices.values():
                    vertex.update_physics()

            for vertex in G.vertices.values():
                vertex.update_logic()
        a.cla()
        G.draw(axis=a)
        a.set_xlim([-12, 12])
        a.set_ylim([-12, 4])
        a.set_zlim([0, 14])

    # G.draw()
    # plt.show()
    anim = animation.FuncAnimation(plt.gcf(), animate, tqdm(range(1000)))
    anim.save("test.mp4", fps=30, dpi=200)
    subprocess.call(["open", "test.mp4"])
