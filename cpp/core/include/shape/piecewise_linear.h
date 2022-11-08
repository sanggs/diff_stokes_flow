#ifndef SHAPE_PIECEWISE_LINEAR_H
#define SHAPE_PIECEWISE_LINEAR_H

#include "shape/parametric_shape.h"

// A piecewise linear curve is defined by 2 control points:
// l(t) = p0 + t(p1 - p0)
// We assume that t from 0 to 1 equals looping over the solid (positive) region in the counter-clockwise order.
class PiecewiseLinear2d : public ParametricShape<2> {
public:
    const real ComputeSignedDistanceAndGradients(const std::array<real, 2>& point,
        std::vector<real>& grad) const override;

private:
    void InitializeCustomizedData() override;
    const Vector2r GetParametricPoint(const real t) const;
    const Vector2r GetParametricDerivative() const;

    Eigen::Matrix<real, 2, 2> control_points_;
    Eigen::Matrix<real, 2, 2> A_;
    Eigen::Matrix<real, 2, 1> B_;
    Eigen::Matrix<real, 2, 2> cA_;
    Eigen::Matrix<real, 2, 1> cB_;

    // Gradients.
    std::array<Eigen::Matrix<real, 2, 2>, 4> cA_gradients_;
    std::array<Eigen::Matrix<real, 2, 1>, 4> cB_gradients_;
};

#endif