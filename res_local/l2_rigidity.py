"""
Mon, Aug 12 2024: 
Code related to my paper on Compressed Sensing x Rigidity Theory
Uses l2 norm instead of the proposed l2/l1 norm to show its limitation
These simulations were carried out in response to the reviewers of my paper
--------------------------------------------------------------------------------
"""

import subprocess, argparse, random
import numpy as np
import cvxpy as cp
from copy import deepcopy
from tqdm import tqdm
from matplotlib import pyplot as plt

from functions import plot, misc, io
from functions.special import rigidity
from config.rigid_network_3D import POSITIONS, EDGES


SAVE_DIR = 'media/l2_centralized/'
JSON_NAME = 'media/Data/l2_data'

# If ANCHORS is not None, then the agents in ANCHORS are assumed to be error-free
# ANCHORS = None
ANCHORS = ['1', '10', '20']

if __name__ == "__main__":
    SIMULATION = {
        1: "single_run",
        2: "error_vs_n_faulty",
        3: "error_vs_scp_iterations",
        4: "error_vs_measurement_error"
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
    PARAM_SET_DEFAULT = {"num_faulty_drones": 6, "num_scp_iterations": 5, 
                         "num_mc": 1, 
                         "slackness_init": 4.0, "slackness_reduction": 3.0, "error_type": "random",
                         "measurement_error": None, "estimation_error": None}

    if SIMULATION[SIMULATION_NO] == "single_run":
        PARAM_SETS.append(deepcopy(PARAM_SET_DEFAULT))
        PARAM_SETS[-1]["num_scp_iterations"] = 6
        PARAM_SETS[-1]["slackness_init"] = 4.0
        PARAM_SETS[-1]["slackness_reduction"] = 3.0
        PARAM_SETS[-1]["num_faulty_drones"] = 1
        
    elif SIMULATION[SIMULATION_NO] == "error_vs_n_faulty":
        for error_type in ["random", "offset"]:
            for num_param_set in range(16):
                PARAM_SETS.append(deepcopy(PARAM_SET_DEFAULT))
                PARAM_SETS[-1]["num_faulty_drones"] = num_param_set + 1
                PARAM_SETS[-1]["num_mc"] = 1000
                PARAM_SETS[-1]["error_type"] = error_type

    
    elif SIMULATION[SIMULATION_NO] == "error_vs_scp_iterations":
        for num_param_set in range(7):
            PARAM_SETS.append(deepcopy(PARAM_SET_DEFAULT))
            PARAM_SETS[-1]["num_scp_iterations"] = num_param_set + 1
            PARAM_SETS[-1]["num_mc"] = 500

    elif SIMULATION[SIMULATION_NO] == "error_vs_measurement_error":
        num_x_vals = 6
        slackness_reduction = [3.0, 2.0, 1.5, 1.5, 1.3, 1.2]
        for estimation_error in np.linspace(0.0, 0.9, 4):
            for i, measurement_error in enumerate(np.linspace(0.0, float(num_x_vals-1), num_x_vals)):
                PARAM_SETS.append(deepcopy(PARAM_SET_DEFAULT))
                PARAM_SETS[-1]["estimation_error"] = estimation_error
                PARAM_SETS[-1]["measurement_error"] = measurement_error
                PARAM_SETS[-1]["num_mc"] = 500
                PARAM_SETS[-1]["slackness_init"] = 4.0 + measurement_error
                PARAM_SETS[-1]["slackness_reduction"] = slackness_reduction[i]

    # ----------------------------------------------------------------------------
    # =============================================================================
    # ----------------------------------------------------------------------------

    if LOAD_DATA and not SIMULATION_NO == 1:
        PARAM_SETS = []

    else:
        # This if-else logic is so wrong, but whatever
        if SIMULATION_NO == 1:
            print("Cannot use 'load' for this. Running new simulation...")
            LOAD_DATA = False
        error_array = {"means": [], "deviations": [], "fail_ratios": [], "detection_success_ratios": []}

    # -------------- LOOP OVER PARAM SETS
    G = rigidity.get_graph(POSITIONS, EDGES)
    if SIMULATION[SIMULATION_NO] == "draw_graph":
        rigidity.draw_only_graph(G)
        plt.show()
        io.exit()

    elif SIMULATION[SIMULATION_NO] == "DEBUG_check_l2l1_condition":
        rigidity.check_l2l1_condition(G)
        io.exit()

    G_est = rigidity.get_graph(POSITIONS, [])
    _ = rigidity.get_distance_rigidity_matrix(G, using_estimates=True, verbose=True)
    true_error_vector = np.empty([len(POSITIONS)*3, 1])

    for params in tqdm(PARAM_SETS):
        errors = []
        failed_to_converge = 0.0
        errors_detected = 0.0

        # ---------- MONTE CARLO LOOP
        for simulation_no in range(params["num_mc"]):
            FAULTY_DRONES = []
            FAULTY_DRONES = [str(number+1) for number in random.sample(range(len(POSITIONS)), params["num_faulty_drones"])]
            # FAULTY_DRONES = ['1']
            if SIMULATION[SIMULATION_NO] == "single_run":
                print("Drones ", FAULTY_DRONES, " are faulty.")
                
            if params["error_type"] == "random":
                BIAS_VECTORS = {name: misc.random_vector_in_box(1.0) for name in FAULTY_DRONES}
            elif params["error_type"] == "offset":
                vec = misc.random_vector_in_box(1.0)
                BIAS_VECTORS = {name: vec for name in FAULTY_DRONES}
            
            true_error_vector[:] = 0.0
            for name in G.vertices:
                G.vertices[name].ekf.x[:3] = G.vertices[name]._pos[:]
                if params["estimation_error"]:
                    G.vertices[name].ekf.x[:3] += misc.random_vector_on_sphere(dimension=3, radius=params["estimation_error"])
                if name in FAULTY_DRONES:
                    G.vertices[name].ekf.x[:3] += BIAS_VECTORS[name]
                true_error_vector[G.index(name)*3:(G.index(name)+1)*3] = G.vertices[name]._pos[:] - G.vertices[name].ekf.x[:3]


            # Initial (biased) estimates are stored in another graph
            for name in G.vertices:
                G_est.vertices[name]._pos = deepcopy(G.vertices[name].ekf.x[0:3])
            
            # Setting up the optimization problem:
            x = cp.Variable(3*len(G.vertices))
            x_k = np.zeros([3*len(G.vertices), 1])

            tol = params["slackness_init"]
            measurements = rigidity.get_phi_D(G, using_estimates=False)
            if params["measurement_error"]:
                measurements += misc.random_vector_on_sphere(dimension=len(EDGES), radius=[params["measurement_error"]]).T[0]

            try:
                for _ in range(params["num_scp_iterations"]):
                    z = measurements - rigidity.get_phi_D(G, using_estimates=True)
                    R = rigidity.get_distance_rigidity_matrix(G, using_estimates=True)
                    objective_function = 0.0
                    for i in range(len(G.vertices)):
                        vec = x_k[i*3:(i+1)*3].T[0] + x[i*3:(i+1)*3]
                        objective_function += vec[0]**2 + vec[1]**2 + vec[2]**2

                    constraints = [cp.atoms.norm(R @ (x) - z) <= tol]
                    if ANCHORS:
                        for a in ANCHORS:
                            constraints.append(cp.atoms.norm(x[3*G.index(a):3*(G.index(a)+1)]) <= 0.0)
                    problem = cp.Problem(cp.Minimize(objective_function), constraints )
                    problem.solve()
                    rigidity.update_estimates(G, x.value)
                    x_k += misc.column(x.value)
                    tol = tol/params["slackness_reduction"]
                
                errors.append(np.linalg.norm(true_error_vector - x_k)/np.linalg.norm(true_error_vector))

                if set(rigidity.get_nonzero_blocks(x_k, tol=0.1)) == set([int(_d)-1 for _d in FAULTY_DRONES]):
                    errors_detected += 1
                # else:
                #     print("These were detected, but not faulty:")
                #     for i in rigidity.get_nonzero_blocks(x_k, tol=0.1):
                #         if i not in [int(_d)-1 for _d in FAULTY_DRONES]:
                #             print(i, x_k[3*i: 3*(i+1)])

                #     print("These were not detected, but faulty:")
                #     for i in set([int(_d)-1 for _d in FAULTY_DRONES]):
                #         if i not in rigidity.get_nonzero_blocks(x_k, tol=0.1):
                #             print(i, x_k[3*i: 3*(i+1)])
                #             print(true_error_vector[3*i: 3*(i+1)])

            except:
                failed_to_converge += 1.0
                continue
        
        # ---------- OUTSIDE MONTE CARLO LOOP
        mean, var = rigidity.get_mean_var(errors)
        error_array["means"].append(mean)
        error_array["deviations"].append(np.sqrt(var))
        error_array["fail_ratios"].append(failed_to_converge/params["num_mc"])
        error_array["detection_success_ratios"].append(errors_detected/(params["num_mc"]))
    
    
    # -------------- OUTSIDE PARAM SET LOOP

    # ---------------------------------------------------------------------------
    # ============================================================================
    # ---------------------------------------------------------------------------

    if LOAD_DATA:
        try:
            data = io.load(JSON_NAME + "_" + str(SIMULATION_NO))
        except:
            print("Error: Could not find saved data for this simulation...")
            io.sys_exit()

        PARAM_SETS = data["PARAM_SETS"]
        error_array = data["error_array"]

    max_fails = max(error_array["fail_ratios"])
    max_fails_ind = error_array["fail_ratios"].index(max_fails)
    print(f"{max_fails*100}% of the simulations of PARAM_SET {max_fails_ind} failed to converge.\nThe rest of the failure %'s are as follows:")   
    print([100*val for val in error_array["fail_ratios"][max_fails_ind:]]) 

    # -------------- PLOTTING
    FIG_WIDTH = 4.5
    APPROX_ERROR_LABEL = r"$\| \mathbf x - \mathbf x^* \|_2 \big/ \|\mathbf x\|_2 $"

    if SIMULATION[SIMULATION_NO] == "single_run":
        plt.figure(figsize=(FIG_WIDTH*0.9, FIG_WIDTH*0.9))
        rigidity.draw_shifted_graph(G, G_est)
        # plot.border3D(plt.gca())
        # plt.gca().set_box_aspect(None, zoom=1.2)

    if SIMULATION[SIMULATION_NO] == "error_vs_n_faulty":    
        x = [_ps["num_faulty_drones"] for _ps in PARAM_SETS]
        y = [_r*100 for _r in error_array["fail_ratios"]]
        plt.plot(x, y, "r--")
        plt.xlabel("No. of Localization Errors")
        plt.ylabel(r"$\%$ of Trials that Failed to Converge")
        plt.savefig(SAVE_DIR + "DEBUG_" + SIMULATION[SIMULATION_NO], 
                    format='png', bbox_inches='tight', dpi=300)

        plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH*0.6))
        x_rand = []
        y_rand = []
        y_off = []
        for i in range(len(PARAM_SETS)):
            if PARAM_SETS[i]["error_type"] == "random":
                x_rand.append(PARAM_SETS[i]["num_faulty_drones"])
                y_rand.append(error_array["detection_success_ratios"][i]*100)
            elif PARAM_SETS[i]["error_type"] == "offset":
                y_off.append(error_array["detection_success_ratios"][i]*100)

        plt.plot(x_rand, y_off, "^--", linewidth=1.0, markersize=4, color="darkred", label="Fully Correlated Errors")
        plt.plot(x_rand, y_rand, ".-", linewidth=1.0, markersize=6, color="black", label="Uncorrelated Errors")
        plt.legend()
        plt.xlabel("No. of Localization Errors")
        plt.ylabel(r"$\%$ of Trials where $\hat{\mathcal D} = \mathcal D$")
        plt.xlim([1, 16])
        plt.ylim([-0.5, 100.5])
        plt.grid(True, linewidth=0.4, alpha=0.6)

    if SIMULATION[SIMULATION_NO] == "error_vs_scp_iterations":
        plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH*0.6))
        x = [_ps["num_scp_iterations"] for _ps in PARAM_SETS]
        y = [_m for _m in error_array["means"]]
        caps = [_d for _d in error_array["deviations"]]
        plt.errorbar(x, y, yerr=caps, fmt='o', markersize=3, capsize=10, color="lightseagreen", label=r"$\pm 1$ Standard Deviation")
        plt.plot(x, y, linewidth=1.0, color="darkslategray")
        plt.xlabel("No. of SCP Iterations")
        plt.ylabel(APPROX_ERROR_LABEL)
        plt.grid(True, linewidth=0.4, alpha=0.6)
        plt.ylim([0, 0.7])
        plt.legend()

    if SIMULATION[SIMULATION_NO] == "error_vs_measurement_error":
        estimation_errors = []
        for params in PARAM_SETS:
            if params["estimation_error"] not in estimation_errors:
                estimation_errors.append(params["estimation_error"])
        
        x_vals = [[] for _ in estimation_errors]
        y_vals = [[] for _ in estimation_errors]
        y_caps = [[] for _ in estimation_errors]
        line_colors = ["seagreen", "darkslategray", "firebrick", "rebeccapurple", "midnightblue"]
        cap_colors = ["mediumaquamarine", "lightseagreen", "salmon", "mediumorchid", "royalblue"]
        linestyles = [".-", "s-", "^-", "v-", "*-"]
        markersizes = [7.8, 3.5, 3.5, 3.5, 5.5]

        for i, params in enumerate(PARAM_SETS):
            line_number = estimation_errors.index(params["estimation_error"])
            x_vals[line_number].append(params["measurement_error"])
            y_vals[line_number].append(error_array["means"][i])
            y_caps[line_number].append(error_array["deviations"][i])

        plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH*0.6))
        for i in range(len(y_vals)):
            # plt.errorbar(x_vals[i], y_vals[i], yerr=y_caps[i], fmt='o', markersize=1, 
            #              capsize=6, color=cap_colors[i], zorder=1.0)
            plt.plot(x_vals[i], y_vals[i], linestyles[i], label=r"$\kappa=$" + f"{estimation_errors[i]:.1f}", 
                     color=line_colors[i], zorder=2.0, markersize=markersizes[i], linewidth=1.0)

        plt.xlabel(r"Measurement Noise ($\epsilon$)")
        plt.ylabel(APPROX_ERROR_LABEL)
        plt.grid(True, linewidth=0.4, alpha=0.6)
        plt.xlim([0, num_x_vals-1])
        plt.ylim([0, 0.8])
        plt.legend()

    if VIEW_FIGURES:
        plot.show()
    
    else:
        save_path = SAVE_DIR + SIMULATION[SIMULATION_NO] + '.png'
        io.clearfile(save_path)
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=500)
        subprocess.run(['open', save_path])

    io.dump(data={"PARAM_SETS": PARAM_SETS, "error_array": error_array}, 
            filename=JSON_NAME + "_" + str(SIMULATION_NO))
