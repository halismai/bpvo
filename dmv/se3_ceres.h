#ifndef BPVO_DMV_SE3_H
#define BPVO_DMV_SE3_H 1
#if defined(WITH_CERES)

#include <ceres/local_parameterization.h>

namespace dmv {

class Se3LocalParameterization : public ceres::LocalParameterization
{
 public:
  static constexpr int NUM_PARAMS = 7;
  static constexpr int DOF = 6;

 public:
  virtual ~Se3LocalParameterization() {}

  virtual bool Plus(const double* x, const double* detla, double* x_plus_delta) const;
  virtual bool ComputeJacobian(const double*, double* jacobian) const;

  virtual int GlobalSize() const { return NUM_PARAMS; }
  virtual int LocalSize() const { return DOF; }

  static void PoseToParams(const double* pose, double* params);

  static void ParamsToPose(const double* params, double* pose);
}; // Se3LocalParameterization

}; // dmv

#endif // WITH_CERES
#endif // BPVO_DMV_SE3_H

