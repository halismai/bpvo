#if defined(WITH_CERES)

#include "dmv/pixel_error.h"

#include <ceres/cost_function.h>
#include <ceres/autodiff_cost_function.h>

namespace bpvo {
namespace dmv {

ceres::CostFunction*
PixelError::Create(const Mat_<double,3,3>& K, const Vec_<double,3>& X,
                   const InterpType& image, double i0)
{
  return new ceres::AutoDiffCostFunction<PixelError, ResidualSize, NumParameters>(
      new PixelError(K, X, image, i0));
}

}
}
#endif
