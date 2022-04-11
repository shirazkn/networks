import numpy as np
import sympy

import functions.worldtime
time = functions.worldtime.time


class NameGenerator:
    """Generates names 'A', 'B', ..."""
    def __init__(self):
        self.counter = 0

    def new_name(self):
        self.counter += 1
        return chr(ord('A')+self.counter-1)

    def generate(self, length):
        names = []
        for _ in range(length):
            names.append(self.new_name())
        return names


class Timer:
    def __init__(self, duration=0.0, randomize=False):
        """
        Everything is in seconds
        """
        self.duration = duration
        self.start_time = time()
        if randomize:
            self.start_time = self.start_time*np.random.rand()

    def reset(self):
        self.start_time = time()

    def get_status_and_reset(self):
        if time() - self.start_time > self.duration:
            self.reset()
            return 1
        return 0

    def get_time_and_reset(self):
        """
        For when you want to use this object as a 'stopwatch'
        """
        try:
            return time() - self.start_time
        finally:
            self.reset()

    def get_elapsed_time(self):
        """
        For when you want to use this object as a 'stopwatch'
        """
        return time() - self.start_time


def column(vector):
    """Returns column vector (np.array) from np.array or list"""
    return np.array(vector, dtype=float).reshape((len(vector), 1))


def tuple_from_col_vec(x):
    return tuple(x.T[0])


def white_noise(cov):
    """
    Generates zero-mean white gaussian noise
    """
    return column(np.random.default_rng().multivariate_normal(mean=np.zeros(len(cov)), cov=cov))


def lemniscate_of_bernoulli(a=1, offset=(0, 0), time_offset=0, scale_y=1):
    """
    Returns the second derivative of the curve (as a function)
    Used for generating nice-looking 2D trajectories
    a is a fixed parameter, t is the parameterization along the curve
    """
    t = sympy.symbols('t')
    t_new = t + time_offset
    x = a*sympy.cos(0.5*t_new)/(1+sympy.sin(0.5*t_new)**2)
    y = scale_y*x*sympy.sin(0.5*t_new)
    x += offset[0]
    y += offset[1]

    x_dot = sympy.simplify(sympy.diff(x, t))
    y_dot = sympy.simplify(sympy.diff(y, t))
    return sympy.utilities.lambdify(t, (x, y, x_dot, y_dot))


def fixed_point_trajectory(point=(0, 0)):
    def trajectory_func(_t):
        return column([point[0], point[1], 0.0, 0.0])
    return trajectory_func
