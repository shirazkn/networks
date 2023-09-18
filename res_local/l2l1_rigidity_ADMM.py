"""
Tue, Aug 8 2023: Code for my paper on ADMM-based Error Recovery
------------------------------------------------------------------------------------
Uses a constrained mixed norm (l2/l1) minimization algorithm to identify localization errors
"""

import subprocess, argparse, random
import numpy as np
import cvxpy as cp
from copy import deepcopy
from tqdm import tqdm
from matplotlib import pyplot as plt

from functions import plot, misc, io
from functions.special import admm, rigidity
from config.rigid_network_3D import POSITIONS, EDGES

SAVE_DIR = 'media/l2l1_ADMM/'
JSON_NAME = 'media/Data/l2l1_ADMM'

plt.rc('text.latex', preamble=r'\usepackage{amssymb}')


if __name__ == "__main__":
    SIMULATION = {
        1: "single_run",
        2: "single_run_all_errors",
        3: "single_run_block_errors",
        4: "single_run_thresholds",
        5: "draw_graph",
    }
    parser = argparse.ArgumentParser(description="Takes simulation specifications as arguments...")
    parser.add_argument("-s", "--simulation", required=True,
                        help = f"Choose simulation number. \nYour choices: {SIMULATION}")
    parser.add_argument("-l", "--load",  action='store_true', required=False,
                        help = f"Loads saved data, rather than running a new simulation.")
    parser.add_argument("-v", "--view",  action='store_true', required=False,
                        help = f"Opens the plot in an interactive window; does not save as an image.")
    
    args = parser.parse_args()
    SIMULATION_NO = int(args.simulation)
    LOAD_DATA = args.load
    VIEW_FIGURES = args.view
    
    PARAM_SETS = []
    PARAM_SET_DEFAULT = {"num_faulty_drones": 6, "num_outer_iterations": 3, 
                         "num_inner_iterations": 10, "num_mc": None}

    if SIMULATION[SIMULATION_NO] == "single_run":
        PARAM_SETS.append(deepcopy(PARAM_SET_DEFAULT))
        PARAM_SETS[-1]["num_mc"] = 1

    elif SIMULATION[SIMULATION_NO] in ["single_run_block_errors", "single_run_thresholds"]:
        PARAM_SETS.append(deepcopy(PARAM_SET_DEFAULT))
        PARAM_SETS[-1]["num_mc"] = 1

    elif SIMULATION[SIMULATION_NO] == "single_run_all_errors":
        PARAM_SETS.append(deepcopy(PARAM_SET_DEFAULT))
        PARAM_SETS[-1]["num_mc"] = 20
        
    elif SIMULATION[SIMULATION_NO] == NotImplemented:
        raise NotImplementedError

    graph = rigidity.get_graph(POSITIONS, [])
    estimates_init = rigidity.get_graph(POSITIONS, [])
    estimates = rigidity.get_graph(POSITIONS, [])

    # --------- MAKE EDGES
    for vtx in estimates.vertices.values():
        vtx.edge_indices = []

    edges = []
    for edge in EDGES:
        edges.append(admm.DistanceMeasurement(edge))
        for _vtx_name in edge:
            estimates.vertices[_vtx_name].edge_indices.append(len(edges) - 1)
            for _nbr_name in edge:
                if not _vtx_name == _nbr_name:
                    graph.add_edge(_vtx_name, _nbr_name)
                    estimates.add_edge(_vtx_name, _nbr_name)
    
    rigidity.get_distance_rigidity_matrix(graph, verbose=True)

    for vtx in estimates.vertices.values():
        vtx.cvx = misc.Namespace(x = cp.Variable((3,1)), 
                                 w = {nbr.name: cp.Variable((3,1)) for nbr in vtx.neighbors})
        vtx.state = misc.Namespace(
            x = np.empty([3, 1]), 
            w = {nbr.name: np.empty([3, 1]) for nbr in vtx.neighbors}, 
            mu = {nbr.name : np.empty([3,1]) for nbr in vtx.neighbors}, 
            lam = {_i : np.empty_like(edges[_i].get_measurement(graph)) for _i in vtx.edge_indices})

    # ========================================================================
    # ----------------- PARAM SETS LOOP

    if SIMULATION[SIMULATION_NO] == "draw_graph":
        rigidity.draw_only_graph(graph)
        plt.show()
        io.exit()
    
    key_list = list(graph.vertices.keys()) + ["all"]
    if LOAD_DATA:
        if SIMULATION[SIMULATION_NO] == "single_run":
            print("Cannot use 'load' for this simulation type. Running new simulation...")
            LOAD_DATA = False
        PARAM_SETS = []

    else:
        error_array = {"means": {v: [] for v in key_list}, "deviations": {v: [] for v in key_list}, 
                       "thresholds": {v: [] for v in graph.vertices}, "detection_success_ratios": []}

    for params in PARAM_SETS:
        errors = {v: [] for v in key_list}
        errors_detected = 0.0

        # ---------------- MONTE CARLO LOOP
        for simulation_no in range(params["num_mc"]):
            FAULTY_DRONES = [vtx_name for vtx_name in random.sample(list(graph.vertices), params["num_faulty_drones"])]
            BIAS_VECTORS = {name: misc.random_vector_in_box(1.0) for name in FAULTY_DRONES}
            
            for vtx_name in estimates_init.vertices:
                estimates_init.vertices[vtx_name]._pos[:3] = graph.vertices[vtx_name]._pos[:]
                if vtx_name in FAULTY_DRONES:
                    estimates_init.vertices[vtx_name]._pos[:3] += BIAS_VECTORS[vtx_name]
                estimates.vertices[vtx_name]._pos[:3] = estimates_init.vertices[vtx_name]._pos[:3]  # ~ Sets x^* to 0

            errors["all"].append(0.0)
            for vtx in estimates.vertices.values():
                errors[vtx.name].append(np.linalg.norm(vtx._pos[:3] - graph.vertices[vtx.name]._pos[:3]))
                errors["all"][-1] += errors[vtx.name][-1]
                admm.reset_dual_variables(vtx)

            # --------------- OUTER LOOP
            rho = 1.0    

            for outer_iteration in tqdm(range(params["num_outer_iterations"]), desc="Simulation No. " + str(simulation_no + 1) + "/" + str(params["num_mc"])):
                z = [edge.get_measurement(graph) - edge.get_measurement(estimates) for edge in edges]
                R = [edge.get_Jacobian(estimates) for edge in edges]

                for vtx in estimates.vertices.values(): 
                    admm.reset_primal_w(vtx)
                
                # --------------- INNER LOOP
                for _ in tqdm(range(params["num_inner_iterations"]), desc="Outer Iteration No. " + str(outer_iteration + 1), leave=False):
                    
                    # --- Primal 1
                    for vtx in estimates.vertices.values():
                        x_star = vtx._pos[:3] - estimates_init.vertices[vtx.name]._pos[:3]
                        objective = cp.norm(x_star + vtx.cvx.x)
                        i = estimates.index(vtx.name)

                        for k in vtx.edge_indices:
                            constraint = R[k][:, 3*i:3*(i+1)] @ vtx.cvx.x - z[k]
                            for nbr in vtx.neighbors:
                                j = estimates.index(nbr.name)
                                constraint += R[k][:, 3*j:3*(j+1)] @ nbr.state.w[vtx.name]

                            objective += ((rho/2.0)*cp.power(cp.norm(constraint), 2)
                                            + vtx.state.lam[k].T @ constraint)
                            
                        for nbr in vtx.neighbors:
                            constraint = vtx.cvx.x - vtx.state.w[nbr.name]
                            objective += ((rho/2.0)*cp.power(cp.norm(constraint), 2) 
                                         + vtx.state.mu[nbr.name].T @ (constraint))

                        # --- Here I compute the 'threshold'
                        if SIMULATION[SIMULATION_NO] == "single_run_thresholds":
                            threshold_vector = np.zeros([3, 1])
                            for k in vtx.edge_indices:
                                threshold_vector += R[k][:, 3*i:3*(i+1)].T @ (R[k][:, 3*i:3*(i+1)] @ (-x_star) - z[k] 
                                                                              + vtx.state.lam[k])
                                for nbr in vtx.neighbors:
                                    j = estimates.index(nbr.name)
                                    threshold_vector += R[k][:, 3*i:3*(i+1)].T @ R[k][:, 3*j:3*(j+1)] @ nbr.state.w[vtx.name]

                            for nbr in vtx.neighbors:
                                threshold_vector += -x_star - vtx.state.w[nbr.name] + vtx.state.mu[nbr.name]

                            error_array["thresholds"][vtx.name].append(np.linalg.norm(threshold_vector))
                        # ---
                        cp.Problem(cp.Minimize(objective), []).solve()
                        admm.update_primal_x(vtx)

                    # --- Primal 2
                    for vtx in estimates.vertices.values():
                        objective = 0.0
                        i = estimates.index(vtx.name)

                        for k in vtx.edge_indices:
                            constraint = R[k][:, 3*i:3*(i+1)] @ vtx.state.x - z[k]
                            for nbr in vtx.neighbors:
                                j = estimates.index(nbr.name)
                                constraint = constraint + R[k][:, 3*j:3*(j+1)] @ nbr.cvx.w[vtx.name]

                            objective += ((rho/2.0)*cp.power(cp.norm(constraint), 2)
                                            + vtx.state.lam[k].T @ constraint)
                            
                        for nbr in vtx.neighbors:
                            constraint = nbr.state.x - nbr.cvx.w[vtx.name]
                            objective += ((rho/2.0)*cp.power(cp.norm(constraint), 2) 
                                            + nbr.state.mu[vtx.name].T @ (constraint))
                        cp.Problem(cp.Minimize(objective), []).solve()
                        admm.update_primal_w(vtx)

                    # --- Multipliers
                    for vtx in estimates.vertices.values():
                        i = estimates.index(vtx.name)

                        for k in vtx.edge_indices:
                            constraint = R[k][:, 3*i:3*(i+1)] @ vtx.state.x - z[k]
                            for nbr in vtx.neighbors:
                                j = estimates.index(nbr.name)
                                constraint += R[k][:, 3*j:3*(j+1)] @ nbr.state.w[vtx.name]

                            vtx.state.lam[k] += rho*(constraint)
                            
                        for nbr in vtx.neighbors:
                            constraint = vtx.state.x - vtx.state.w[nbr.name]
                            vtx.state.mu[nbr.name] += rho*(constraint) 

                    errors["all"].append(0.0)
                    for vtx in estimates.vertices.values():
                        errors[vtx.name].append(np.linalg.norm(vtx._pos[:3] + vtx.state.x - graph.vertices[vtx.name]._pos[:3]))
                        errors["all"][-1] += errors[vtx.name][-1]
                    # END INNER LOOP ------------------
                
                for vtx in estimates.vertices.values():
                    admm.update_estimates(vtx)
                # END OUTER LOOP ------------------

            non_zero_blocks = []
            for vtx in estimates.vertices.values():
                if errors[vtx.name][-1] > 0.1:
                    non_zero_blocks.append(vtx.name)
            if set(non_zero_blocks) == set(FAULTY_DRONES):
                errors_detected += 1.0
            # END MONTE CARLO LOOP ------------------

        for v in key_list:
            error_array["means"][v], error_array["deviations"][v] \
                = admm.get_mean_dev(errors[v], params["num_mc"])
            if params["num_mc"] == 1:
                error_array["deviations"][v] = []

        error_array["detection_success_ratios"].append(errors_detected/(params["num_mc"]))
    # END PARAM SET LOOP ------------------
    # ============================================================================

    if LOAD_DATA:
        try:
            data = io.load(JSON_NAME + "_" + str(SIMULATION_NO))
        except:
            print("Error: Could not find saved data for this simulation...")
            io.sys_exit()

        PARAM_SETS = data["PARAM_SETS"]
        FAULTY_DRONES = data["FAULTY_DRONES"]
        error_array = data["error_array"]
    
    io.dump(data={"PARAM_SETS": PARAM_SETS, "error_array": error_array, "FAULTY_DRONES": FAULTY_DRONES}, 
            filename=JSON_NAME + "_" + str(SIMULATION_NO))

    # -------------- PLOTTING
    FIG_WIDTH = 4.5
    if SIMULATION[SIMULATION_NO] == "single_run":
        plt.figure(figsize=(FIG_WIDTH*0.9, FIG_WIDTH*0.9))
        rigidity.draw_shifted_graph_ADMM(graph, estimates_init, estimates)


    if SIMULATION[SIMULATION_NO] == "single_run_all_errors":

        # All errors combined
        plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH*0.6))
        APPROX_ERROR_LABEL = r"$\| \mathbf x - \mathbf x^* \|$" 

        for v in ["all"]:
            if error_array["deviations"][v]:
                # plt.errorbar(range(len(error_array["means"][v])), error_array["means"][v], yerr=error_array["deviations"][v], 
                #             fmt='o', markersize=3, capsize=10, color="lightseagreen", label="$\pm 1$ Standard Deviation")
                admm.plot_std_patch(error_array["means"][v], error_array["deviations"][v])
            plt.plot(error_array["means"][v], '.-', linewidth=0.8, markersize=2.0, label="Average", color="black")

        plt.xlabel("No. of Inner Iterations")
        plt.ylabel(APPROX_ERROR_LABEL)
        plt.grid(True, linewidth=0.4, alpha=0.6)
        plt.ylim([0,7.0])
        plt.xlim([0, len(error_array["means"][v])-1])

    if SIMULATION[SIMULATION_NO] == "single_run_block_errors":
        # plt.figure(figsize=(FIG_WIDTH*0.9, FIG_WIDTH*0.9))
        # rigidity.draw_shifted_graph_ADMM(graph, estimates_init, estimates)
        # plot.show()

        # All errors combined
        plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH*0.6))
        APPROX_ERROR_LABEL = r"$\| \mathbf x[i] - \mathbf x^*[i] \|$"
        YLIM = 0.8
        
        for v in graph.vertices:
            color = "red" if v in FAULTY_DRONES else "darkslategray"
            plt.plot(error_array["means"][v], linewidth=1.0, color=color)

        for i in range(1, PARAM_SETS[-1]["num_outer_iterations"]):
            inner_len = PARAM_SETS[-1]["num_inner_iterations"]
            h_loc = int(inner_len*i)
            plt.plot([h_loc, h_loc], [0, YLIM], linewidth=0.5, color='black')
        
        plt.plot([], linewidth=1.0, color="red", label="$i \in \mathcal D$")
        plt.plot([], linewidth=1.0, color="darkslategray", label="$i \in \mathcal D^\complement$")

        plt.xlabel("No. of Inner Iterations")
        plt.ylabel(APPROX_ERROR_LABEL)
        plt.grid(True, linewidth=0.4, alpha=0.6)
        plt.ylim([0, YLIM])
        plt.xlim([0, len(error_array["means"][v])-1])
        plt.legend()

    if SIMULATION[SIMULATION_NO] == "single_run_thresholds":
        # All errors combined
        plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH*0.6))
        YLIM = 25.0
        
        for v in graph.vertices:
            color = "red" if v in FAULTY_DRONES else "darkslategray"
            plt.plot(error_array["thresholds"][v], linewidth=1.0, color=color)

        for i in range(1, PARAM_SETS[-1]["num_outer_iterations"]):
            inner_len = PARAM_SETS[-1]["num_inner_iterations"]
            h_loc = int(inner_len*i)
            plt.plot([h_loc, h_loc], [0, YLIM], linewidth=0.5, color='black')
        
        plt.plot([], linewidth=1.0, color="red", label="$i \in \mathcal D$")
        plt.plot([], linewidth=1.0, color="darkslategray", label="$i \in \mathcal D^\complement$")

        plt.plot([0, len(error_array["means"][v])], [1.0, 1.0], 'b--')

        plt.xlabel("No. of Inner Iterations")
        plt.ylabel("$\|A_i^{\intercal} b_i\|$")
        plt.grid(True, linewidth=0.4, alpha=0.6)
        plt.ylim([0, YLIM])
        plt.xlim([0, len(error_array["means"][v])-2])
        plt.legend()


    if VIEW_FIGURES:
        plot.show()
    
    else:
        save_path = SAVE_DIR + SIMULATION[SIMULATION_NO] + '.png'
        io.clearfile(save_path)
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=500)
        subprocess.run(['open', save_path])

