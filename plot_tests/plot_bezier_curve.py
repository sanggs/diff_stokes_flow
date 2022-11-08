import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc

def initialize_bezier_curve(bounds, n = 4):
    x_bounds = bounds[0]
    assert(len(x_bounds) == 2)
    assert(x_bounds[1] > x_bounds[0])
    y_bounds = bounds[1]
    assert(len(y_bounds) == 2)
    assert(y_bounds[1] > y_bounds[0])

    cx = np.linspace(x_bounds[0], x_bounds[1], n) # uniformly distributed knots
    cy = np.random.rand(n)
    cy = cy[:] * (y_bounds[1] - y_bounds[0]) + y_bounds[0]

    control_points = np.zeros((n, 2))
    control_points[:, 0] = cx
    control_points[:, 1] = cy
    return control_points
    
