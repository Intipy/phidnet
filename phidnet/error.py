import numpy as np



def mean_squared_error(y, t):   # Mean squared error
    return 0.5 * np.sum((y-t)**2)



def cross_entropy_error(y, t):   # Cross entropy error
    h = 1e-7
    return -np.sum(t * np.log(y + h))