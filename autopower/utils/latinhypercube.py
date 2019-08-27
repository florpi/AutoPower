from pyDOE import lhs
import numpy as np

def latin_hypercube(n_pts, mins, maxs):
    """
    Returns the n_pts number of samples in a latin hypercube.
    """
    #return a latin_hypercube
    design = lhs(np.size(maxs), samples=n_pts)
    for i in range(2):
        design[:, i] = design[:, i] * (maxs[i]-mins[i]) + mins[i]
    return design

