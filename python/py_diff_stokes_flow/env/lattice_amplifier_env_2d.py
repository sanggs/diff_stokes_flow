import numpy as np

from py_diff_stokes_flow.env.env_base import EnvBase
from py_diff_stokes_flow.common.common import ndarray

class LatticeAmplifierEnv2d(EnvBase):
    def __init__(self, seed, folder):
        np.random.seed(seed)

        cell_nums = (10, 10)
        E = 100
        nu = 0.499
        vol_tol = 1e-3
        edge_sample_num = 2
        EnvBase.__init__(self, cell_nums, E, nu, vol_tol, edge_sample_num, folder)
        self._lattice_scale = 2
        self._lattice_cell_nums = (int(cell_nums[0]/self._lattice_scale), int(cell_nums[1]/self._lattice_scale))

        # Initialize the parametric shapes.
        self._parametric_shape_info = [ ('bezier', 8), ('bezier', 8) ]
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
        self._interface_boundary_type = 'free-slip'

        # Other data members.
        self._inlet_velocity = inlet_velocity
        self._outlet_velocity = 3 * inlet_velocity
        self._inlet_range = inlet_range
        self._initialize_lattice_nodes()

    def _initialize_lattice_nodes(self):
        lx, ly = self._lattice_cell_nums
        num_lattice_nodes = (lx+1)*(ly+1)
        self._lattice_nodes = np.zeros((num_lattice_nodes * 2, 1))
        lattice_dx = 1./float(lx)

        for ci in range(lx):
            for cj in range(ly):
                for ni in range(2):
                    for nj in range(2):
                        li = (ci + ni) * (ly+1) + (cj + nj)
                        self._lattice_nodes[2 * li + 0] = (ci + ni)
                        self._lattice_nodes[2 * li + 1] = (cj + nj)

    def _embed_control_points_in_lattice(self, points):
        cx, cy = self._cell_nums
        lx, ly = self._lattice_cell_nums
        assert(lx == ly)

        lattice_dx = 1./float(lx)
        lattice_dy = 1./float(ly)
        pts = np.copy(points)
        pts = np.reshape(pts, (2, -1, 2))
        pts[:, :, 0] *= (lx/cx)
        pts[:, :, 1] *= (ly/cy)
        (n0, n1, n2) = pts.shape

        num_lattice_nodes = (lx+1)*(ly+1)
        self._lattice_weight_matrix  = np.zeros((n0 * n1 * n2, num_lattice_nodes * 2))

        for j in range(n0):
            for i in range(1, n1-1):
                pt = np.floor(pts[j][i])
                pt = pt.astype(np.int)

                x1 = pt[0]
                x2 = pt[0] + 1
                y1 = pt[1]
                y2 = pt[1] + 1
                x = pts[j][i][0]
                y = pts[j][i][1]

                for ni in range(2):
                    for nj in range(2):
                        col_Idx = (pt[0] + ni) * (ly+1) + (pt[1] + nj)
                        row_Idx = j * (n1 * n2) + i * n2
                        u = (x2 - x) if ni == 0 else (x - x1)
                        v = (y2 - y) if nj == 0 else (y - y1)
                        self._lattice_weight_matrix[row_Idx + 0, 2 * col_Idx + 0] = u * v
                        self._lattice_weight_matrix[row_Idx + 1, 2 * col_Idx + 1] = u * v
        
        pt = np.floor(pts[0][0])
        pt = pt.astype(np.int)
        for i in range(2):
            if (pt[i] == self._lattice_cell_nums[i]):
                pt[i] -= 1
        for nj in range(2):
            ni = 1

            y1 = pt[1]
            y2 = pt[1] + 1
            y = pts[0][0][1]

            col_Idx = (pt[0] + ni) * (ly+1) + (pt[1] + nj)
            row_Idx = 0

            v = (y2 - y) if nj == 0 else (y - y1)
            self._lattice_weight_matrix[row_Idx + 1, 2 * col_Idx + 1] = v

        pt = np.floor(pts[1][3])
        pt = pt.astype(np.int)
        for i in range(2):
            if (pt[i] == self._lattice_cell_nums[i]):
                pt[i] -= 1
        for nj in range(2):
            ni = 1

            y1 = pt[1]
            y2 = pt[1] + 1
            y = pts[1][3][1]

            col_Idx = (pt[0] + ni) * (ly+1) + (pt[1] + nj)
            row_Idx = 14
            v = (y2 - y) if nj == 0 else (y - y1)
            self._lattice_weight_matrix[row_Idx + 1, 2 * col_Idx + 1] = v

        self._lattice_weight_matrix *= (cx/lx)

        # np.savetxt('mat.txt' , self._lattice_weight_matrix)
        # print(np.reshape(points, (2, -1, 2)))
        # print(np.matmul(self._lattice_weight_matrix,self._lattice_nodes))
    
    def _lattice_to_shape_params(self, lattice_nodes, prt=False):
        p = np.matmul(self._lattice_weight_matrix, lattice_nodes)
        self._embed_control_points_in_lattice(p)
        
        p[0] = 10.0
        p[7] = 1.25
        p[9] = 8.75
        p[14] = 10.0
        params = ndarray(np.concatenate([p.ravel()]))
        if (prt):
            print(params)
        
        return ndarray(params).copy(), self._lattice_weight_matrix.copy()
    
    def _lattice_and_weights_to_shape_params(self, lattice_nodes, ew, prt=False):
        p = np.matmul(ew, lattice_nodes)
        
        p[0] = 10.0
        p[7] = 1.25
        p[9] = 8.75
        p[14] = 10.0
        params = ndarray(np.concatenate([p.ravel()]))
        if (prt):
            print(params)
        
        return ndarray(params).copy(), self._lattice_weight_matrix.copy()

    def _deform_lattice(self, dx):
        assert(dx.shape == self._lattice_nodes.shape)
        return np.copy(self._lattice_nodes) + dx

    def _variables_to_shape_params(self, x):
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
        params = np.concatenate([lower.ravel(), upper.ravel()])

        # Jacobian.
        J = np.zeros((params.size, x.size))
        J[1, 4] = 1
        J[2, 2] = 1
        J[3, 3] = 1
        J[4, 0] = 1
        J[5, 1] = 1
        J[10, 0] = 1
        J[11, 1] = -1
        J[12, 2] = 1
        J[13, 3] = -1
        J[15, 4] = -1
        # Scale it by cx and cy.
        J[:, 0] *= cx
        J[:, 2] *= cx
        J[:, 1] *= cy
        J[:, 3] *= cy
        J[:, 4] *= cy
        
        return ndarray(params).copy(), ndarray(self._lattice_weight_matrix).copy()

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