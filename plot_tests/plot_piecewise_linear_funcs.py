
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.collections as mc

def initialize_control_points(bounds, nc = 4, seed = 1.):
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

def plot_curves(curves, colors, labels = None):
    fig, ax = plt.subplots()
    for j in range(len(curves)):
        curve = curves[j]
        segs = []
        for i in range(len(curve)-1):
            x1 = curve[i+1]
            x0 = curve[i]
            segs.append((x0, x1))
        ax.add_collection(mc.LineCollection(segs, colors=colors[j]))      
    return plt

def get_line(c0, c1):
    m = (c1[1] - c0[1])/(c1[0] - c0[0])
    c = c0[1] - m * c0[0]
    return m, c

def verify(point, control_points, i, t):
    ct = np.copy(control_points)
    c0 = ct[i]
    c1 = ct[i+1]
    m, c = get_line(c0, c1)
    assert(c0[1] - m * c0[0] - c == 0)
    ds = (point[1] - m * point[0] - c )/np.sqrt(1 + m * m)
    ds = np.abs(ds)

    C = np.array([ct[i], ct[i+1]])
    C = np.transpose(C)
    A = np.array([[1, -1], [0, 1]])
    tvec = np.array([1, t])
    proj = np.matmul(np.matmul(C, A), tvec)
    dist = np.linalg.norm(proj - point)

    d = c1 - c0
    t1 = (d[0] * point[0] + d[1] * point[1] - d[0] * c0[0] - d[1] * c0[1])/(d[0] * d[0] + d[1] * d[1])
    proj1 = c0 + t1 * d
    tvec1 = np.array([1, t1])
    proj2 = np.matmul(np.matmul(C, A), tvec1)

    # print(np.abs(ds - dist), t, t1, proj, proj1, i, m, c, c0, c1)
    assert(np.abs(ds - dist) < 1e-11)
    assert(np.abs(t - t1) < 1e-11)
    assert(proj1[0] == proj2[0] and proj2[1] == proj1[1])

def get_signed_distance(point, curves):
    global_dist = float("inf")
    signed_dist = 0.0
    actual_t = float("inf")
    pt = np.array([point[0], point[1]])
    min_i = -10
    control_points = []
    for c in curves:
        for i in range(len(c)-1):
            control_points.append([c[i], c[i+1]])
    control_points = np.array(control_points)
    
    ct = np.copy(control_points)
    num_lines = len(control_points)
    for i in range(num_lines):
        C = np.array(ct[i])
        C = np.transpose(C)
        A = np.array([[1, -1], [0, 1]])
        B = np.array([-1, 1])

        product = np.matmul(np.transpose(np.matmul(C, A)), np.matmul(C, B))
        assert(product[1] >= 0)
        bcp = np.matmul(np.transpose(np.matmul(C, B)), pt)
        possible_t = (bcp - product[0])/product[1]

        ts = [0, possible_t, 1] if (possible_t >= 0 and possible_t <= 1) else [0, 1]
        min_dist = float("inf")
        min_t = float("inf")
        for t in ts:
            tvec = np.array([1, t])
            proj = np.matmul(np.matmul(C, A), tvec)
            dist = np.linalg.norm(proj - pt)
            if (dist < min_dist):
                min_t = t
                min_dist = dist
        assert(min_t >= 0 and min_t <= 1)

        t_final = [1, min_t]
        proj = np.matmul(np.matmul(C, A), t_final)
        q = proj - pt
        min_tangent = np.matmul(C, B)
        z = q[0] * min_tangent[1] - q[1] * min_tangent[0]
        sign = -1 if z <= 0 else 1
        
        if (min_dist < global_dist):
            global_dist = min_dist
            signed_dist = sign * global_dist
            min_i = i
            actual_t = possible_t
    assert(min_i >= 0)
    # if (point[0] == 3 and point[1] == 0):
    #     print(signed_dist)
    #     verify(point, control_points, min_i, actual_t)

    return signed_dist

def plot_curve_points(curves, cx, cy):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    padding = 5
    ax.set_title('Bezier curve plot')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-padding, cx + padding])
    ax.set_ylim([-padding, cy + padding])
    ax.set_aspect('equal')
    ax.axis('off')

    # Plot cells.
    lines = []
    colors = []
    shift = 0.0
    for i in range(cx):
        for j in range(cy):
            color = 'k'
            pts = [(i + shift, j + shift),
                (i + 1 - shift, j + shift),
                (i + 1 - shift, j + 1 - shift),
                (i + shift, j + 1 - shift)
            ]
            lines += [
                (pts[0], pts[1]),
                (pts[1], pts[2]),
                (pts[2], pts[3]),
                (pts[3], pts[0])
            ]
            colors += [color,] * 4
    ax.add_collection(mc.LineCollection(lines, colors=colors, linewidth=0.5))

    nodes = []
    dists = []
    colors = []
    for i in range(cx+1):
        for j in range(cy+1):
            nodes.append([i, j])
            dists.append(get_signed_distance(np.array([i, j]), curves))
            if (dists[-1] <= 0):
                colors.append('red')
            else:
                colors.append('blue')
            # print('dist at [{}, {}] = {}'.format(i, j, dists[-1]))
    nodes = np.array(nodes)
    ax.scatter(nodes[:, 0], nodes[:, 1], color=colors)

    # Plot solid-fluid interfaces.
    lines = []
    def cutoff(d0, d1):
        assert d0 * d1 <= 0
        return -d0 / (d1 - d0)
    for i in range(cx):
        for j in range(cy):
            ps = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            ds = [get_signed_distance(p, curves) for p in ps]
            vs = []
            ps = np.array(ps)
            for k in range(4):
                k_next = (k + 1) % 4
                if ds[k] * ds[k_next] <= 0:
                    t = cutoff(ds[k], ds[k_next])
                    vs.append((1 - t) * ps[k] + t * ps[k_next])
            vs_len = len(vs)
            for k in range(vs_len):
                lines.append((vs[k], vs[(k + 1) % vs_len]))
    ax.add_collection(mc.LineCollection(lines, colors='tab:orange', linewidth=1))

    #Plot end points of control points
    # ax.plot(control_points[:, 0], control_points[:, 1], '*--', color='red')

def linear_interpolate(p0, p1, t):
    return p0 + t * (p1 - p0)

def get_point_on_bezier_curve(control_points, t):
    num_cp = len(control_points)
    assert(num_cp == 4) # assume cubic bezier
    ct = np.copy(control_points)
    g0 = linear_interpolate(ct[0], ct[1], t)
    g1 = linear_interpolate(ct[1], ct[2], t)
    g2 = linear_interpolate(ct[2], ct[3], t)
    b0 = linear_interpolate(g0, g1, t)
    b1 = linear_interpolate(g1, g2, t)
    p = linear_interpolate(b0, b1, t)
    return p

def divide_bezier_curve(cubic_bezier_control_points, num_dof):
    dx = 1./float(num_dof - 2)
    ratios = np.linspace(dx, 1.-dx, num_dof-2)
    points = np.zeros((num_dof, 2))
    points[0] = cubic_bezier_control_points[0]
    points[-1] = cubic_bezier_control_points[-1]
    for i in range(num_dof-2):
        points[i+1] = get_point_on_bezier_curve(cubic_bezier_control_points, ratios[i])    
    return points

cx = 6
cy = 6
num_dof = 7

lower_bounds = [[0, 1], [0.15, 0.35]]
lower_bounds[0][0] *= cx
lower_bounds[1][0] *= cy
lower_bounds[0][1] *= cx
lower_bounds[1][1] *= cy

upper_bounds = [[0, 1], [0.65, 0.85]]
upper_bounds[0][0] *= cx
upper_bounds[1][0] *= cy
upper_bounds[0][1] *= cx
upper_bounds[1][1] *= cy

lower_controls = initialize_control_points(lower_bounds, 4, 1.0)
upper_controls = initialize_control_points(upper_bounds, 4, 2.0)

lower_curve = divide_bezier_curve(lower_controls, num_dof)
upper_curve = divide_bezier_curve(upper_controls, num_dof)

# lower_curve = initialize_control_points(lower_bounds, 10, 1.0)
# upper_curve = initialize_control_points(upper_bounds, 10, 2.0)

lower_curve_rev = np.zeros_like(lower_curve)
for i in range(len(lower_curve)):
    lower_curve_rev[len(lower_curve) - i - 1] = lower_curve[i]

curves = [lower_curve_rev, upper_curve]

plot_curve_points(curves, cx, cy)
plt.show()
