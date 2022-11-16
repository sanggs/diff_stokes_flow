import numpy as np

from py_diff_stokes_flow.env.env_base import EnvBase
from py_diff_stokes_flow.common.common import ndarray

from matplotlib import pyplot as plt
import matplotlib.collections as mc

class PiecewiseControls2d(EnvBase):
    def __init__(self, seed, folder):
        np.random.seed(seed)

        cell_nums = (20, 20)
        E = 100
        nu = 0.499
        vol_tol = 1e-3
        edge_sample_num = 2
        EnvBase.__init__(self, cell_nums, E, nu, vol_tol, edge_sample_num, folder)
        # self._lattice_scale = 2
        # self._lattice_cell_nums = (int(cell_nums[0]/self._lattice_scale), int(cell_nums[1]/self._lattice_scale))

        self._num_pieces_upper = 10
        self._num_pieces_lower = 10

        # Initialize the parametric shapes.
        shape_info = ('piecewise_linear', 4)
        self._parametric_shape_info = []
        for _ in range(self._num_pieces_upper+self._num_pieces_lower):
            self._parametric_shape_info.append(shape_info)
        
        # Initialize the node conditions.
        self._node_boundary_info = []
        inlet_velocity = 1
        inlet_range = ndarray([0.125, 0.875])
        inlet_lb, inlet_ub = inlet_range * cell_nums[1]
        for j in range(cell_nums[1] + 1):
            if inlet_lb < j < inlet_ub:
                self._node_boundary_info.append(((0, j, 0), inlet_velocity))
                self._node_boundary_info.append(((0, j, 1), 0))
        # Initialize the interface.
        # print(self._node_boundary_info)
        self._interface_boundary_type = 'free-slip'

        # Other data members.
        self._inlet_velocity = inlet_velocity
        self._outlet_velocity = 3 * inlet_velocity
        self._inlet_range = inlet_range
        self._point_shape_mat = None
        self._initialize_points_to_shapes()

    def _initialize_points_to_shapes(self):
        num_shapes_dof = self._num_pieces_lower + self._num_pieces_upper
        num_shapes_dof *= 4
        
        num_points = self._num_pieces_lower + 1 + self._num_pieces_upper + 1
        num_points *= 2

        self._point_shape_mat = np.zeros((num_points, num_shapes_dof))
        for i in range(0, self._num_pieces_lower):
            for j in range(4):
                self._point_shape_mat[2*i+j, 4*i+j] = 1.0
                
        co = self._num_pieces_lower * 4
        ro = 2 * self._num_pieces_lower + 2 
        for i in range(0, self._num_pieces_upper):
            for j in range(4):
                self._point_shape_mat[ro + 2*i+j, co + 4*i+j] = 1.0
    
    def _reset_fixed_params(self, p):
        cx, cy = self._cell_nums
        
        p = np.reshape(p, (2, -1, 2))

        p[0][0][0] = 1.0 * cx
        p[0][-1][1] = self._inlet_range[0] * cy
        p[1][0][1] = self._inlet_range[1] * cy
        p[1][-1][0] = 1.0 * cx

        p = p.ravel()

    def _reset_fixed_curve_params(self, c):
        cx, cy = self._cell_nums
        sh = c.shape
        c = np.reshape(c, (-1, 2))
        
        c[0][0] = 1.0*cx
        c[2 * self._num_pieces_lower - 1][1] = self._inlet_range[0] * cy
        c[2 * self._num_pieces_lower][1] = self._inlet_range[1] * cy
        c[2 * self._num_pieces_lower + 2 * self._num_pieces_upper - 1][0] = 1.0 * cx

        c = np.reshape(c, sh)

    def _lattice_to_shape_params(self, points):
        curves = np.matmul(np.transpose(self._point_shape_mat), points)
        self._reset_fixed_curve_params(curves)
        J = np.transpose(self._point_shape_mat)
        return curves, J.copy()

    def _initialize_bezier_curve_controls(self, x):
        x = ndarray(x).copy().ravel()
        assert x.size == 5
        
        cx, cy = self._cell_nums
        # Convert x to the shape parameters.
        lower = ndarray([
            [1, x[4]],
            x[2:4],
            x[:2],
            [0, self._inlet_range[0]],
        ])
        lower[:, 0] *= cx
        lower[:, 1] *= cy
        upper = ndarray([
            [0, self._inlet_range[1]],
            [x[0], 1 - x[1]],
            [x[2], 1 - x[3]],
            [1, 1 - x[4]],
        ])
        upper[:, 0] *= cx
        upper[:, 1] *= cy
        
        return upper, lower

    def _linear_interpolate(self, p0, p1, t):
        return p0 + t * (p1 - p0)

    def _get_point_on_bezier_curve(self, control_points, t):
        num_cp = len(control_points)
        assert(num_cp == 4) # assume cubic bezier
        ct = np.copy(control_points)
        g0 = self._linear_interpolate(ct[0], ct[1], t)
        g1 = self._linear_interpolate(ct[1], ct[2], t)
        g2 = self._linear_interpolate(ct[2], ct[3], t)
        b0 = self._linear_interpolate(g0, g1, t)
        b1 = self._linear_interpolate(g1, g2, t)
        p = self._linear_interpolate(b0, b1, t)
        return p

    def _divide_bezier_curve(self, cubic_bezier_control_points, num_dof):
        dx = 1./float(num_dof - 2)
        ratios = np.linspace(dx, 1.-dx, num_dof-2)
        points = np.zeros((num_dof, 2))
        points[0] = np.copy(cubic_bezier_control_points[0])
        points[-1] = np.copy(cubic_bezier_control_points[-1])
        for i in range(num_dof-2):
            points[i+1] = self._get_point_on_bezier_curve(cubic_bezier_control_points, ratios[i])    
        return points

    def _initialize_control_points(self, bounds, nc = 4, seed = 1):
        np.random.seed(int(seed))
        x_bounds = bounds[0]
        assert(len(x_bounds) == 2)
        assert(x_bounds[1] > x_bounds[0])
        y_bounds = bounds[1]
        assert(len(y_bounds) == 2)
        assert(y_bounds[1] > y_bounds[0])

        cx = np.linspace(x_bounds[0], x_bounds[1], nc) # uniformly distributed knots
        cy = np.random.rand(nc)
        cy = cy[:] * (y_bounds[1] - y_bounds[0]) + y_bounds[0]

        control_points = np.zeros((nc, 2))
        control_points[:, 0] = cx
        control_points[:, 1] = cy
        return control_points

    def _variables_to_shape_params(self, x):
        upper, lower = self._initialize_bezier_curve_controls(x)
        lower_dof = self._divide_bezier_curve(lower, self._num_pieces_lower+1)
        upper_dof = self._divide_bezier_curve(upper, self._num_pieces_upper+1)

        self._lattice_nodes = np.concatenate([lower_dof.ravel(), upper_dof.ravel()])

        lower_curve = np.zeros((2 * (len(lower_dof) - 1), 2))
        upper_curve = np.zeros((2 * (len(upper_dof) - 1), 2))

        for i in range(len(lower_dof)-1):
            lower_curve[(2 * i)] = lower_dof[i]
            lower_curve[(2 * i + 1)] = lower_dof[i+1]
        
        for i in range(len(upper_dof)-1):
            upper_curve[(2 * i)] = upper_dof[i]
            upper_curve[(2 * i + 1)] = upper_dof[i+1]

        params = np.concatenate([lower_curve.ravel(), upper_curve.ravel()])

        J = self._point_shape_mat
        return ndarray(params).copy(), J.copy()

    def _loss_and_grad_on_velocity_field(self, u):
        u_field = self.reshape_velocity_field(u)
        grad = np.zeros(u_field.shape)
        cnt = 0
        loss = 0
        for j, (ux, uy) in enumerate(u_field[-1]):
            if ux > 0:
                cnt += 1
                u_diff = ndarray([ux, uy]) - ndarray([self._outlet_velocity, 0])
                loss += u_diff.dot(u_diff)
                grad[-1, j] += 2 * u_diff
        loss /= cnt
        grad /= cnt
        return loss, ndarray(grad).ravel()

    def sample(self):
        return np.random.uniform(low=self.lower_bound(), high=self.upper_bound())

    def _embedding_weights(self):
        return ndarray(self._lattice_weight_matrix).copy()

    def lower_bound(self):
        return ndarray([0.16, 0.05, 0.49, 0.05, 0.05])

    def upper_bound(self):
        return ndarray([0.49, 0.49, 0.83, 0.49, 0.49])
    