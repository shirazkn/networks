import numpy as np


class NameGenerator:
    """Generates names 'A', 'B', ..."""
    def __init__(self):
        self.counter = 0

    def new_name(self):
        self.counter += 1
        return chr(ord('A')+self.counter-1)

    def generate(self, length):
        names = []
        for i in range(length):
            names.append(self.new_name())
        return names


def column(vector):
    """Returns column vector (np.array) from np.array or list"""
    return np.array(vector).reshape((len(vector), 1))
