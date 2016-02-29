#if defined(WITH_CERES)

#include "dmv/reprojection_error.h"
#include "dmv/se3.h"

#include <ceres/autodiff_cost_function.h>

using namespace Eigen;

namespace dmv {

ReprojectionErrorSe3::ReprojectionErrorSe3(const double* K, const double* X,
                                           const double* x)
 : _K(K), _X(X), _x(x) {}

template <class T>
bool ReprojectionErrorSe3::operator()(const T* const params, T* residual) const
{
  const Map<const Se3_<T>> se3(params);
  const Mat44_<T> m = se3.matrix();
  const Mat33_<T> R = m.template block<3,3>(0,0);
  const Vec3_<T>  t = m.template block<3,1>(0,3);

  const Map<const Mat33_<double>> K(_K);
  const Map<const Vec3_<double>> X(_X);
  const Map<const Vec2_<double>> x(_x);

  const Vec3_<T> uvw = K.template cast<T>() * (R * X.template cast<T>() + t);
  const Vec2_<T> xp = uvw.template head<2>() * (1.0 / uvw[2]);

  Map<Vec2_<T>> err(residual);
  err = x.template cast<T>() - xp;

  return true;
}

ceres::CostFunction*
ReprojectionErrorSe3::Create(const double* K, const double* X, const double* x)
{
  return new ceres::AutoDiffCostFunction<ReprojectionErrorSe3,2,7>(
      new ReprojectionErrorSe3(K, X, x));
}

}; // dmv


#endif
