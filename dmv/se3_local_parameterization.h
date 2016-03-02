#ifndef BPVO_DMV_SE3_LOCAL_PARAMETERIZATION_H
#define BPVO_DMV_SE3_LOCAL_PARAMETERIZATION_H

#if defined(WITH_CERES) && defined(WITH_SOPHUS)

#include <dmv/lie_group_local_parameterization.h>
#include <sophus/se3.hpp>

namespace bpvo {
namespace dmv {

template <typename T> using Se3_ = Sophus::SE3Group<T>;
template <typename T> using Se3LocalParameterization_ = LieGroupLocalParameterization<Se3_<T>>;

typedef Se3LocalParameterization_<double> Se3LocalParameterization;

}; // dmv
}; // bpvo

#endif
#endif // BPVO_DMV_SE3_LOCAL_PARAMETERIZATION_H
