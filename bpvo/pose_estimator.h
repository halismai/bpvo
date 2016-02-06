#ifndef BPVO_POSE_ESTIMATOR_H
#define BPVO_POSE_ESTIMATOR_H

#include <bpvo/types.h>
#include <bpvo/debug.h>
#include <bpvo/mestimator.h>

#include <limits>
#include <cmath>
#include <iostream>

#include <Eigen/Cholesky> // for ldlt()

namespace bpvo {

struct PoseEstimatorParameters
{
  int maxIterations = 50;
  float functionTolerance   = 1e-6;
  float parameterTolerance  = 1e-6;
  float gradientTolerance   = 1e-6;
  LossFunctionType lossFunction = LossFunctionType::kHuber;

  VerbosityType verbosity = VerbosityType::kSilent;

  /**
   */
  inline PoseEstimatorParameters() {}

  /**
   */
  explicit PoseEstimatorParameters(const AlgorithmParameters& p);

  /**
   */
  void relaxTolerance(int max_it = 42, float scale_by = 10.0f);
}; // PoseEstimatorParameters


template <class TemplateDataT, class SystemBuilderT>
class PoseEstimator
{
 public:
  typedef typename TemplateDataT::Channels Channels;
  typedef typename SystemBuilderT::Hessian Hessian;
  typedef typename SystemBuilderT::Gradient Gradient;
  typedef typename SystemBuilderT::Gradient ParameterVector;

 public:
  /**
   */
  PoseEstimator(AlgorithmParameters p = AlgorithmParameters())
      : _params(p), _scale_estimator(1e-3) {}

  inline void setParameters(const PoseEstimatorParameters& p) {
    _params = p;
  }

  inline const PoseEstimatorParameters& parameters() const { return _params; }

  /**
   */
  OptimizerStatistics run(TemplateDataT*, const Channels&, Matrix44& T);

 protected:
  PoseEstimatorParameters _params;
  AutoScaleEstimator _scale_estimator;
  std::vector<float> _residuals;
  std::vector<float> _weights;
  std::vector<uint8_t> _valid;

 protected:
  float runIteration(TemplateDataT*, const Channels&, const Matrix44& T,
                     Hessian* A, Gradient* g);
}; // PoseEstimator


template <class TemplateDataT, class SystemBuilderT> inline
OptimizerStatistics PoseEstimator<TemplateDataT, SystemBuilderT>::
run(TemplateDataT* tdata, const Channels& channels, Matrix44& T)
{
  static const char* FMT_STR = "   %3d      %13.6g  %12.3g    %12.6g   %12.6g\n";
  static constexpr float sqrt_eps = std::sqrt(std::numeric_limits<float>::epsilon());

  _scale_estimator.reset(); // reset the scale for a new run

  OptimizerStatistics ret;
  ret.numIterations = 0;
  ret.firstOrderOptimality = 0;
  ret.status = PoseEstimationStatus::kMaxIterations;

  //
  // first iteration outside to set the first-order optimality threshold and see
  // if we are at the solution
  //
  Hessian A;
  Gradient b;
  ParameterVector dp;

  float F_norm = runIteration(tdata, channels, T, &A, &b);
  float G_norm = b.template lpNorm<Eigen::Infinity>();
  float g_tol = _params.gradientTolerance * std::max(G_norm, sqrt_eps);
  float f_tol = _params.functionTolerance;
  float p_tol = _params.parameterTolerance;

  ret.firstOrderOptimality = G_norm;
  ret.finalError = F_norm;

  if(G_norm < g_tol) {
    printf("Initial value is optimal\n");
    ret.status = PoseEstimationStatus::kGradientTolReached;
    return ret;
  }

  if(VerbosityType::kIteration == _params.verbosity) {
    printf("\n");
    printf(" Iteration    Residuals          |G|         step norm       delta error\n");
    printf(FMT_STR, 0, F_norm, G_norm, 0.0f, 0.0f);
  }

  float dp_norm_prev= 0.0f;
  float f_norm_prev = F_norm;
  bool converged = false;

  while(ret.numIterations++ < _params.maxIterations) {

    dp = A.ldlt().solve(b);

    if(!(A * dp).isApprox(b)) {
      Warn("Solver failed\n");
      ret.status = PoseEstimationStatus::kSolverError;
      return ret;
    }

    // TODO: LM style check if error is not increasing
    Matrix44 T_est = tdata->warp().paramsToPose(-dp);
    T = T * T_est;

    if(converged)
      break;

    //
    // test convergence based on the parameters and gradient
    //
    float dp_norm = dp.norm();
    if(dp_norm < p_tol || dp_norm < p_tol * ( sqrt_eps + dp_norm_prev )) {
      ret.status = PoseEstimationStatus::kParameterTolReached;
      converged = true;
    }

    //
    // re-linearize
    //
    F_norm = runIteration(tdata, channels, T, &A, &b);
    G_norm = b.template lpNorm<Eigen::Infinity>();

    if(G_norm < g_tol) {
      ret.status = PoseEstimationStatus::kGradientTolReached;
      converged = true; // this way we do the update at the top of the loop
    }

    float delta_error = fabs(F_norm - f_norm_prev);
    if(!converged) {
      if(F_norm < f_tol || F_norm < f_tol * (sqrt_eps + f_norm_prev) ||
         delta_error < f_tol * (sqrt_eps + f_norm_prev)) {
        ret.status = PoseEstimationStatus::kFunctionTolReached;
        converged = true;
      }
    }

    if(VerbosityType::kIteration == _params.verbosity) {
      printf(FMT_STR, ret.numIterations, F_norm, G_norm, dp_norm, delta_error);
    }

    dp_norm_prev = dp_norm;
    f_norm_prev = F_norm;
  }

  ret.finalError = F_norm;
  ret.firstOrderOptimality = G_norm;

  return ret;
}

template <class TemplateDataT, class SystemBuilderT> inline
float PoseEstimator<TemplateDataT, SystemBuilderT>::
runIteration(TemplateDataT* tdata, const Channels& channels, const Matrix44& pose,
             Hessian* H, Gradient* G)
{
  tdata->computeResiduals(channels, pose, _residuals, _valid);
  auto sigma = _scale_estimator.estimateScale(_residuals, _valid);
  computeWeights(_params.lossFunction, _residuals, _valid, sigma, _weights);

  return SystemBuilderT::Run(tdata->jacobians(), _residuals, _weights, _valid, H, G);
}

}; // bpvo

#endif // BPVO_POSE_ESTIMATOR_H

