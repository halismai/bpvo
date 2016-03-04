#if defined(WITH_CERES)

#include "dmv/patch_error.h"

#include <ceres/cost_function.h>
#include <ceres/autodiff_cost_function.h>

namespace bpvo {
namespace dmv {

ceres::CostFunction*
PatchError::Create(const Mat_<double,3,3>& K, const Vec_<double,3>& X,
                   const InterpType& image, const double* i0)
{
  return new ceres::AutoDiffCostFunction<PatchError, ResidualSize, NumParameters>(
      new PatchError(K, X, image, i0));
}

}
}

#endif
