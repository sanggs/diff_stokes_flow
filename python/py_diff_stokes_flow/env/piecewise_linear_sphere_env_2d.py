import numpy as np

from py_diff_stokes_flow.env.env_base import EnvBase
from py_diff_stokes_flow.common.common import ndarray

from matplotlib import pyplot as plt
import matplotlib.collections as mc

class PiecewiseLinearSphereEnv2d(EnvBase):
    def __init__(self, seed, folder, cell_dim = (20, 20), youngs_modulus = 100,
                 poissons_ratio = 0.499, volume_tolerance = 1e-3, inlets = [[0.125, 0.875]], inlet_velocities = [[1., 0.]],
                 num_pieces = [6, 6], bounds = [[0.16, 0.05, 0.49, 0.05, 0.05], [0.49, 0.49, 0.83, 0.49, 0.49]]):
        np.random.seed(seed)

        assert(len(num_pieces) % 2 == 0)
        assert(int(len(num_pieces)/2) == len(inlets))
        assert(len(bounds) == len(num_pieces))

        cell_nums = cell_dim
        E = youngs_modulus
        nu = poissons_ratio
        vol_tol = volume_tolerance
        edge_sample_num = 2
        EnvBase.__init__(self, cell_nums, E, nu, vol_tol, edge_sample_num, folder)
        
        self._num_pieces = num_pieces
        self._num_total_pieces = 0
        for i in self._num_pieces:
            self._num_total_pieces += i
        print("Num total pieces = {}".format(self._num_total_pieces))

        # # Initialize the parametric shapes.
        shape_info = ('piecewise_linear', 4)
        self._parametric_shape_info = []
        for _ in range(self._num_total_pieces):
            self._parametric_shape_info.append(shape_info)
        
        # Initialize the node conditions.
        self._node_boundary_info = []
        for i, inlet_range in enumerate(inlets):
            inlet_velocity = inlet_velocities[i]
            inlet_range = ndarray(inlet_range)
            inlet_lb, inlet_ub = inlet_range * cell_nums[1]
            # print(i, inlet_range, inlet_velocity)
            for j in range(cell_nums[1] + 1):
                if inlet_lb < j < inlet_ub:
                    # print("Setting {} as inlet with velocity = {}".format(j, inlet_velocity))
                    self._node_boundary_info.append(((0, j, 0), inlet_velocity[0]))
                    self._node_boundary_info.append(((0, j, 1), inlet_velocity[1]))
        # Initialize the interface.
        # print(self._node_boundary_info)
        # self._interface_boundary_type = 'free-slip'

        self._bounds = bounds

        # Other data members.
        self._inlet_velocity = inlet_velocity
        self._inlet_range = inlets
        self._point_shape_mat = None

    def _initialize_bezier_curve_controls(self, x, i):
        num_regions = int(len(self._num_pieces)/2)
        y_min = (float(i)/float(num_regions))

        x = ndarray(x).copy().ravel()
        assert x.size == 5
        
        cx, cy = self._cell_nums
        # Convert x to the shape parameters.
        lower = ndarray([
            [1, self._inlet_range[i][0]],
            # [1, x[4]],
            x[2:4],
            x[:2],
            [0, self._inlet_range[i][0]],
        ])

        lower[:, 0] *= cx
        lower[:, 1] *= cy * (float(1)/float(num_regions))
        lower[:, 1] += cy * y_min
        upper = ndarray([
            [0, self._inlet_range[i][1]],
            [x[0], 1 - x[1]],
            [x[2], 1 - x[3]],
            # [1, 1 - x[4]],
            [1, self._inlet_range[i][1]]
        ])
        upper[:, 0] *= cx
        upper[:, 1] *= cy * (float(1)/float(num_regions)) 
        upper[:, 1] += cy * y_min

        # reset the inlets 
        upper[0, 1] = self._inlet_range[i][1] * cy
        upper[-1, 1] = self._inlet_range[i][1] * cy
        lower[-1, 1] = self._inlet_range[i][0] * cy
        lower[0, 1] = self._inlet_range[i][0] * cy

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

    def _embed_sphere_inside_bbox(self, points):
        min_c = np.array([np.min(points[:, 0]), np.min(points[:, 1])])
        max_c = np.array([np.max(points[:, 0]), np.max(points[:, 1])])
        center = (max_c + min_c)/2.
        center[0] += np.random.rand() * 2
        dists = np.sqrt((points[:, 0] - center[0])**2 + (points[:, 1] - center[1])**2) 
        min_dist = np.min(dists)
        if (min_dist > 2):
            radius = 0.5 + np.random.rand() * 0.75
            return [center[0], center[1], radius]
        return None

    def _variables_to_shape_params(self, list_x, add_noise = False):
        if (len(self._parametric_shape_info) > 0):
            shape_info = ('piecewise_linear', 4)
            self._parametric_shape_info = []
            for _ in range(self._num_total_pieces):
                self._parametric_shape_info.append(shape_info)
        params = []
        sphere_params = []
        for i, x in enumerate(list_x):
            upper, lower = self._initialize_bezier_curve_controls(x, i)

            lower_dof = self._divide_bezier_curve(lower, self._num_pieces[2 * i + 0] + 1)
            upper_dof = self._divide_bezier_curve(upper, self._num_pieces[2 * i + 1] + 1)

            if (add_noise):
                noise_l = np.random.rand(self._num_pieces[2 * i + 0] + 1)
                noise_u = np.random.rand(self._num_pieces[2 * i + 1] + 1)
                noise_u[0] = 0.
                noise_u[-1] = 0.
                noise_l[0] = 0.
                noise_l[-1] = 0.
                lower_dof[:, 1] += noise_l
                upper_dof[:, 1] += noise_u

            dofs = np.zeros([len(lower_dof) + len(upper_dof), 2])
            dofs[0:len(lower_dof), :] = lower_dof[:, :]
            dofs[len(lower_dof):] = upper_dof[:, :]

            lower_curve = np.zeros((2 * (len(lower_dof) - 1), 2))
            upper_curve = np.zeros((2 * (len(upper_dof) - 1), 2))

            for i in range(len(lower_dof)-1):
                lower_curve[(2 * i)] = lower_dof[i]
                lower_curve[(2 * i + 1)] = lower_dof[i+1]
            
            for i in range(len(upper_dof)-1):
                upper_curve[(2 * i)] = upper_dof[i]
                upper_curve[(2 * i + 1)] = upper_dof[i+1]
            
            params.append(np.concatenate([lower_curve.ravel(), upper_curve.ravel()]))
            # if(lenparams):
            #     params = np.concatenate([lower_curve.ravel(), upper_curve.ravel()])
            # else:
            #     params = np.concatenate(params, np.concatenate([lower_curve.ravel(), upper_curve.ravel()]))

            res = self._embed_sphere_inside_bbox(dofs)
            if (res is not None):
                shape_info = ('sphere', 3) # num params = dim + 1
                self._parametric_shape_info.append(shape_info)
                sphere_params.append(np.array(res))

        return_params = ndarray(params).copy().ravel()
        return_params = list(return_params)
        for sp in sphere_params:
            for spi in sp:
                return_params.append(spi)
        return ndarray(return_params).copy(), None

    def _loss_and_grad_on_velocity_field(self, u):
        # u_field = self.reshape_velocity_field(u)
        # grad = np.zeros(u_field.shape)
        # cnt = 0
        # loss = 0
        # for j, (ux, uy) in enumerate(u_field[-1]):
        #     if ux > 0:
        #         cnt += 1
        #         u_diff = ndarray([ux, uy]) - ndarray([self._outlet_velocity, 0])
        #         loss += u_diff.dot(u_diff)
        #         grad[-1, j] += 2 * u_diff
        # loss /= cnt
        # grad /= cnt
        return NotImplementedError

    def sample(self):
        regions = int(len(self._num_pieces)/2)
        samples = []
        for i in range(regions):
            x = np.random.uniform(low=self.lower_bound(2 * i), high=self.upper_bound(2 * i + 1))
            samples.append(np.copy(x))
        return samples

    def lower_bound(self, i):
        return ndarray(self._bounds[int(i)])

    def upper_bound(self, i):
        return ndarray(self._bounds[int(i)])
