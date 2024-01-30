#include "shape/piecewise_linear.h"
#include "common/common.h"
#include "unsupported/Eigen/Polynomials"

void PiecewiseLinear2d::InitializeCustomizedData() {
    CheckError(param_num() == 4, "Inconsistent number of parameters.");
    
    control_points_ = Eigen::Map<const Eigen::Matrix<real, 2, 2>>(params().data(), 2, 2);
    // std::cout << control_points_ << std::endl;
    A_ << 1, -1, 0, 1;
    B_ << -1, 1;
    cA_ = control_points_ * A_;
    cB_ = control_points_ * B_;

    // Gradients.
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
            const int idx = j * 2 + i;
            cA_gradients_[idx].setZero();
            cA_gradients_[idx].row(i) = A_.row(j);
        }
}

const real PiecewiseLinear2d::ComputeSignedDistanceAndGradients(const std::array<real, 2>& point,
    std::vector<real>& grad) const {
    const Vector2r p(point[0], point[1]);
    
    Eigen::Matrix<real, 2, 1> C = cA_.transpose() * cB_;
    assert(C(1) >= 0);
    real ptcb = cB_.transpose() * p;
    real possible_t = (ptcb - C(0))/C(1);
    //assert(possible_t >= 0 || possible_t <= 1);
    if (possible_t < 0.0 || possible_t > 1.0)
        possible_t = (real)0;

    // Add two end points - this concludes the candidate set.
    std::vector<real> ts{ 0, possible_t, 1 };
    // for (const real& t : ts_full)
    //     if (0 <= t && t <= 1)
    //         ts.push_back(t);

    // Pick the minimal distance among them.
    real min_dist = std::numeric_limits<real>::infinity();
    real min_t = 0;
    Vector2r min_proj(0, 0);
    for (const real t : ts) {
        Vector2r proj = GetParametricPoint(t);
        const real dist = (proj - p).norm();
        if (dist < min_dist) {
            min_dist = dist;
            min_proj = proj;
            min_t = t;
        }
    }

    // Determine the sign.
    const Vector2r min_tangent = GetParametricDerivative();
    if (min_tangent.norm() == 0) {
        std::cout << "The Parametric curve is singular. Control points are probably duplicated." << std::endl;
        std::cout << control_points_ << std::endl;
        CheckError(false, "Singular Parametric curve.");
    }
    const Vector2r q = min_proj - p;
    // Consider the sign of q x min_tangent: positive = interior.
    const real z = q.x() * min_tangent.y() - q.y() * min_tangent.x(); // cross product
    const real sign = z >= 0 ? 1.0 : -1.0;

    // if (point[1] == (real)4)
    //     std::cout << "[" << p[0] << "," << p[1] << "] " << z << " " << sign << " " << min_dist << std::endl; 

    // Compute the gradient.
    // control_point -> coeff -> min_t -> min_proj -> min_dist.
    // According to the envelope theorem, we can safely assume min_t does not change during the gradient computation.
    // min_proj = GetParametricPoint(t) = cA_ * ts.
    
    const Vector2r min_ts(1, min_t);
    Eigen::Matrix<real, 2, 4> min_proj_gradients; min_proj_gradients.setZero();
    for (int i = 0; i < 4; ++i) {
        min_proj_gradients.col(i) = cA_gradients_[i] * min_ts;
    }
    // min_proj -> min_dist: min_dist = |min_proj - p|.
    const real eps = Epsilon();
    Vector2r q_unit = Vector2r::Zero();
    if (min_dist > eps) q_unit = q / min_dist;
    const Vector4r grad_vec(q_unit.transpose() * min_proj_gradients);
    grad.resize(4);
    for (int i = 0; i < 4; ++i) grad[i] = sign * grad_vec(i);
    return sign * min_dist;
}

const Vector2r PiecewiseLinear2d::GetParametricPoint(const real t) const {
    const Vector2r ts(1, t);
    return cA_ * ts;
}

const Vector2r PiecewiseLinear2d::GetParametricDerivative() const {
    return cB_;
}
