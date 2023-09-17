"""
Keeps track of various configuration options for the simulation

Note: Do not do `from functions.worldtime import time` as that (potentially??) creates a new instance of
WorldTime() each time its called! Use `import functions.worldtime`'` instead
"""
from config.constants import WORLD_TIMESTEP


class WorldTime:
    def __init__(self):
        self.time = 0.0
        self.dt = WORLD_TIMESTEP
        print("New WorldTime object was created! Make sure there's only one of these...\n")
        # TODO ^This can be checked automatically by counting instances

    def step(self):
        self.time += self.dt

    def get_time(self):
        return self.time

    def get_functions(self):
        # This is a hack! Keeps the WorldTime() object out of namespace so that it doesn't get imported everywhere
        return self.get_time, self.step


time, step = WorldTime().get_functions()