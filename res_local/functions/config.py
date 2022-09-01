# Plotting
PLOT_LIM = 15
OFFSET = [0, 0]
PLOT_TRAJECTORIES = False  # Todo: Bring 'plot estimates' here
MARKER_TYPE = None
# GPS_SYMBOL_OFFSET = [0.12, 0.12]
GPS_SYMBOL_OFFSET = [0, 0]
COLORS = {"dark_green": "#00A400", "dark_red": "#D80000"}

# Sensors
GPS_TIMEOUT = 1.0/2.0
INS_TIMEOUT = 1.0/10.0

# Simulation
WORLD_TIMESTEP = 0.02
ODE_STEPS = 5
SPEEDUP_FACTOR = 1.0

# Speed everything up...
WORLD_TIMESTEP *= SPEEDUP_FACTOR
ODE_TIMESTEP = WORLD_TIMESTEP / (ODE_STEPS+0.001)
GPS_TIMEOUT *= SPEEDUP_FACTOR
INS_TIMEOUT *= SPEEDUP_FACTOR
