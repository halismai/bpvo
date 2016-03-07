#ifndef BPVO_DMV_LIE_GROUP_LOCAL_PARAMETERIZATION_H
#define BPVO_DMV_LIE_GROUP_LOCAL_PARAMETERIZATION_H

#if defined(WITH_CERES)

#include <Eigen/Core>
#include <ceres/local_parameterization.h>

namespace bpvo {
namespace dmv {

template <class GroupType>
class LieGroupLocalParameterization : public ceres::LocalParameterization
{
  typedef GroupType Group;
  typedef typename Group::Scalar Scalar;
  static constexpr int NumParameters = Group::num_parameters;
  static constexpr int DoF           = Group::DoF;

  typedef Eigen::Matrix<Scalar,DoF,1> Vector;
  typedef Eigen::Matrix<Scalar,DoF,NumParameters> Jacobian;

 public:
  virtual ~LieGroupLocalParameterization() {}

  virtual bool Plus(const Scalar* G, const Scalar* delta, Scalar* G_ret_) const
  {
    using namespace Eigen;

    Map<Group> G_ret(G_ret_);
    G_ret = Map<const Group>(G) * Group::exp(Map<const Vector>(delta));

    return true;
  }

  virtual bool ComputeJacobian(const Scalar* G_, Scalar* J_) const
  {
    using namespace Eigen;

    Map<Jacobian> J(J_);
    J = Map<const Group>(G_).internalJacobian().transpose();

    return true;
  }

  inline int GlobalSize() const { return NumParameters; }
  inline int LocalSize() const { return DoF; }
}; // LieGroupLocalParameterization

}; // dmv
}; // bpvo

#endif

#endif // BPVO_DMV_LIE_GROUP_LOCAL_PARAMETERIZATION_H
