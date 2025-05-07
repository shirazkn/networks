import numpy as np
import sympy
from copy import deepcopy

import functions.worldtime
time = functions.worldtime.time


class NameGenerator:
    """Generates names 'A', 'B', ..."""
    def __init__(self, name_type='letter'):
        if name_type == 'number':
            self.new_name = self.new_name_number
        elif name_type == 'letter':
            self.new_name = self.new_name_letter
        else:
            raise ValueError
        self.counter = 0

    def new_name_letter(self):
        self.counter += 1
        return chr(ord('A')+self.counter-1)

    def new_name_number(self):
        self.counter += 1
        return str(self.counter)

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
            self.start_time = self.start_time - np.random.rand()*duration

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


class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

            
def column(vector):
    """Returns column vector (np.array) from np.array or list"""
    return np.array(vector, dtype=float).reshape((len(vector), 1))


def tuple_from_col_vec(x):
    return tuple(x.T[0])


def random_gaussian(cov):
    """
    Samples from a zero-mean Gaussian random variable
    """
    return column(np.random.default_rng().multivariate_normal(mean=np.zeros(len(cov)), cov=cov))


def random_vector_in_box(a):
    """
    Generates (uniform randomly) a 3D position vector in the cube
    of side length 2a centered at the origin
    """
    vec = 2*a*np.random.default_rng().random([3,1])
    vec = vec - column([a, a, a])
    # if (vec.T @ vec) > 0.075:
    return vec
    # else:
    #     # print(vec, "is too close to the origin")
    #     return random_vector_in_box(a)


def random_vector_on_sphere(dimension, radius):
    gaussian_vector = random_gaussian(cov=np.identity(dimension))
    gaussian_vector /= np.linalg.norm(gaussian_vector)
    return radius*gaussian_vector


def rotation_matrix(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    return np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])


def lemniscate_of_bernoulli(a=1, offset=(0, 0), time_offset=0, scale_y=1):
    """
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

    if len(offset)==2:
        return sympy.utilities.lambdify(t, (x, y, x_dot, y_dot))
    else:
        return sympy.utilities.lambdify(t, (x, y, offset[2], x_dot, y_dot, 0.0))


def fixed_point_trajectory(point=(0, 0)):
    """
    For when the vehicle should 'hold' its position
    """
    if len(point) == 2:
        def trajectory_func(_t):
            return column([point[0], point[1], 0.0, 0.0])
    else:
        def trajectory_func(_t):
            return column([point[0], point[1], point[2], 0.0, 0.0, 0.0])
    return trajectory_func
