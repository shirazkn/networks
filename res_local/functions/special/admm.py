import numpy as np
from functions import misc
from matplotlib import pyplot as plt


MEASUREMENT_TYPES = {
    "distance": {
        "dimension": 1
    },
    "bearing3D": {
        "dimension": 3
    }
}
    

class HyperEdge:
    def __init__(self, vertices):
        self.vertices = vertices
        self.size = len(self.vertices)


class DistanceMeasurement(HyperEdge):
    # Uses the ._pos of whatever graph its fed to compute the measurement and its Jacobian
    def __init__(self, vertices):
        super().__init__(vertices)

    def get_measurement(self, G):
        edge = G.vertices[self.vertices[0]]._pos[:3] - G.vertices[self.vertices[1]]._pos[:3]
        return 0.5*(edge.T @ edge)

    def get_Jacobian(self, G):
        edge = G.vertices[self.vertices[0]]._pos[:3] - G.vertices[self.vertices[1]]._pos[:3]
        indices = [G.index(vtx) for vtx in self.vertices]
        R_k = np.zeros([1, 3*len(G.vertices)])
        R_k[:, 3*indices[0]:3*(indices[0]+1)] = edge.T
        R_k[:, 3*indices[1]:3*(indices[1]+1)] = -1*edge.T
        return R_k        


def get_mean_dev(vec, num_mc):
    # vec ~ [iter 1, iter 2, ..., iter N, iter N+1, iter N+2, ..., iter N * num_mc] 
    # where N = int(len(vec)/num_mc)
    means_length = int(len(vec)/num_mc)

    means = []
    sums_of_squares = []
    for i in range(means_length):
        things_to_average = []
        for j in range(num_mc):
            things_to_average.append(vec[j*means_length+i])

        means.append(np.mean(things_to_average))
        sums_of_squares.append(np.sum([(things_to_average[k] - means[-1])**2 for k in range(num_mc)])) 

    return means, [np.sqrt(val/(num_mc-1)) for val in sums_of_squares]

def update_primal_x(vtx):
    vtx.state.x = misc.column(vtx.cvx.x.value)

def update_primal_w(vtx):
    for nbr in vtx.neighbors:
        nbr.state.w[vtx.name] = misc.column(nbr.cvx.w[vtx.name].value)

def update_estimates(vtx):
    # vtx should be the vertex of a (virtual) 'estimates' graph which is used for plotting
    vtx._pos[:3] += vtx.state.x

def reset_primal_w(vtx):
    for var in vtx.state.w.values():
        var[:] = 0.0

def reset_dual_variables(vtx):
    for var in vtx.state.mu.values():
        var[:] = 0.0
    for var in vtx.state.lam.values():
        var[:] = 0.0
            
def plot_std_patch(means, deviations):
    np_means = np.array(means)
    np_deviations = np.array(deviations)
    plt.fill_between(range(len(np_means)), np_means - np_deviations, np_means + np_deviations,
                     color='slategray', alpha=0.3, label=r"$\pm 1$ Standard Deviation")