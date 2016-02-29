#if defined(WITH_CERES)

#include <sophus/se3.hpp>
#include <Eigen/Core>

#include "dmv/se3.h"
#include "dmv/se3_ceres.h"

using namespace Eigen;

namespace  dmv {

bool Se3LocalParameterization::Plus(const double* x, const double* delta,
                                    double* x_plus_delta) const
{
  const Map<const Se3_<double>> T(x);
  const Map<const Se3_<double>::Tangent> theta(delta);

  Map<Se3_<double>> T_ret(x_plus_delta);
  T_ret = T * Se3_<double>::exp(theta);

  return true;
}

bool Se3LocalParameterization::ComputeJacobian(const double* x,
                                               double* jacobian) const
{
  const Map<const Se3_<double>> T(x);
  Map<Matrix<double,6,7>> J(jacobian);

  J = T.internalJacobian().transpose();

  return true;
}

void Se3LocalParameterization::PoseToParams(const double* T, double* p)
{
  memcpy(p, Se3_<double>( Mat44_<double>(T) ).data(),
         NUM_PARAMS * sizeof(double));
}

void Se3LocalParameterization::ParamsToPose(const double* /*p*/, double* /*T*/)
{
  // TODO
}

}; // dmv

#endif

