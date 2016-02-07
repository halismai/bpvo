#ifndef BPVO_POSE_ESTIMATOR_BASE_H
#define BPVO_POSE_ESTIMATOR_BASE_H

#include <bpvo/types.h>
#include <bpvo/pose_estimator_params.h>
#include <bpvo/mestimator.h>

#include <cmath>
#include <limits>
#include <vector>

namespace bpvo {

template <class TDataT> class PoseEstimatorGN;
template <class TDataT> class PoseEstimatorLM;


template <class> class PoseEstimatorTraits;

template<> template <class TDataT>
class PoseEstimatorTraits< PoseEstimatorGN<TDataT> >
{
 public:
  typedef TDataT TemplateData;
  typedef typename TemplateData::Warp Warp;
  typedef typename TemplateData::Jacobian Jacobian;
  typedef typename TemplateData::Channels Channels;

  static constexpr int NumParameters = Jacobian::ColsAtCompileTime;
  typedef typename Jacobian::Scalar DataType;

  typedef Eigen::Matrix<DataType, NumParameters, 1> Gradient;
  typedef Eigen::Matrix<DataType, NumParameters, 1> ParameterVector;
  typedef Eigen::Matrix<DataType, NumParameters, NumParameters> Hessian;
}; // PoseEstimatorTraits

template <int N>
struct PoseEstimatorData_
{
  Eigen::Matrix<float,N,N> H;
  Eigen::Matrix<float,N,1> G;
  Eigen::Matrix<float,N,1> dp;
  Matrix44 T;

  inline float gradientNorm() const {
    return G.template lpNorm<Eigen::Infinity>();
  }

  inline bool solve()
  {
    dp = H.ldlt().solve(G);
    return (H*dp).isApprox(G);
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}; // PoseEstimatorData


template <class Derived>
class PoseEstimatorBase
{
 public:
  typedef PoseEstimatorTraits<Derived> Triats;

  typedef typename PoseEstimatorTraits<Derived>::TemplateData TemplateData;
  typedef typename TemplateData::Channels Channels;
  typedef typename PoseEstimatorTraits<Derived>::Warp         Warp;
  typedef typename PoseEstimatorTraits<Derived>::Jacobian     Jacobian;
  typedef typename PoseEstimatorTraits<Derived>::Gradient     Gradient;
  typedef typename PoseEstimatorTraits<Derived>::ParameterVector ParameterVector;
  typedef typename PoseEstimatorTraits<Derived>::Hessian         Hessian;

  static constexpr int NumParameters = Triats::NumParameters;

  typedef std::vector<float> ResidualVector;
  typedef std::vector<float> WeightsVector;
  typedef std::vector<uint8_t> ValidVector;

  typedef PoseEstimatorData_<NumParameters> PoseEstimatorData;

 public:
  /**
   * \param data the template/reference frame data
   * \param cn   the input channels
   * \param T    Pose (input and output). It is used for initialization as well
   *             as the return value
   */
  OptimizerStatistics run(TemplateData* data, const Channels& cn, Matrix44& T);

  inline void setParameters(const PoseEstimatorParameters& p) {
    _params = p;
  }

  inline const PoseEstimatorParameters& parameters() const { return _p; }

 protected:
  PoseEstimatorParameters _params;
  AutoScaleEstimator _scale_estimator;

  ResidualVector _residuals;
  WeightsVector _weights;
  ValidVector _valid;

  float _f_norm_prev = 0.0f;
  float _g_tol = 0.0f;
  int _num_fun_evals = 0;

  inline const Derived* derived() const { return static_cast<const Derived*>(this); }
  inline       Derived* derived()       { return static_cast<Derived*>(this); }

 protected:
  inline const ResidualVector& residuals() const { return _residuals; }
  inline       ResidualVector& residuals()       { return _residuals; }

  inline const WeightsVector& weights() const { return _weights; }
  inline       WeightsVector& weights()       { return _weights; }

  inline const ValidVector& valid() const { return _valid;  }
  inline       ValidVector& valid()       { return _valid;  }

 protected:
  static constexpr const char* _verbose_fmt_str_first_it =
      " %5d       %5d   %13.6g    %12.3g\n";
  static constexpr const char* _verbose_fmt_str =
      " %5d       %5d   %13.6g    %12.3g    %12.6g\n";

  void printHeader(float f_val, float g_norm) const
  {
    if(_params.verbosity == VerbosityType::kDebug ||
       _params.verbosity == VerbosityType::kIteration) {
      printf("\n                                        First-Order         Norm of \n"
             " Iteration  Func-count    Residual       optimality            step\n");
      fprintf(stdout, _verbose_fmt_str_first_it, 0, _num_fun_evals, f_val, g_norm);
    }
  }

  void printIteration(int iteration, float f_val, float g_norm, float dp_norm) const
  {
    if(_params.verbosity == VerbosityType::kDebug ||
       _params.verbosity == VerbosityType::kIteration) {
      fprintf(stdout, _verbose_fmt_str, iteration, _num_fun_evals, f_val, g_norm, dp_norm);
    }
  }

  bool testConvergence(float dp_norm, float dp_norm_prev, float g_norm, float f_norm,
                       PoseEstimationStatus& status) const
  {
    const auto sqrt_eps = std::sqrt(std::numeric_limits<float>::epsilon());

    if(dp_norm < _params.parameterTolerance ||
       dp_norm < _params.parameterTolerance * (sqrt_eps + dp_norm_prev)) {
      status = PoseEstimationStatus::kParameterTolReached;
      return true;
    }

    if(f_norm < _params.functionTolerance ||
       f_norm < _params.functionTolerance * (sqrt_eps + _f_norm_prev)) {
      status = PoseEstimationStatus::kFunctionTolReached;
      return true;
    }

    if(g_norm < _g_tol) {
      status = PoseEstimationStatus::kGradientTolReached;
      return true;
    }

    return false;
  }

  inline void reset()
  {
    _scale_estimator.reset();
    _f_norm_prev = 0.0f;
    _g_tol = 0.0f;
    _num_fun_evals = 0;
  }

  inline void printResult(const OptimizerStatistics& s) const
  {
    fprintf(stdout, "PoseEstimator: %d iters |F|=%g |G|=%g term reason: %s\n",
            s.numIterations, s.finalError, s.firstOrderOptimality,
            ToString(s.status).c_str());
  }
}; // PoseEstimatorBase


template <class Derived> inline
OptimizerStatistics PoseEstimatorBase<Derived>::
run(TemplateData* tdata, const Channels& channels, Matrix44& T)
{
  this->reset();

  OptimizerStatistics ret;
  ret.numIterations = 0;
  ret.firstOrderOptimality = 0;
  ret.status = PoseEstimationStatus::kMaxIterations;

  PoseEstimatorData data;
  data.T = T;

  float f_norm = derived()->linearize(tdata, channels, data),
        g_norm = data.gradientNorm();

  this->_g_tol = _params.gradientTolerance *
      std::max(g_norm, std::sqrt(std::numeric_limits<float>::epsilon())),

  this->printHeader(f_norm, g_norm);

  if(g_norm < _g_tol) {
    ret.status = PoseEstimationStatus::kGradientTolReached;
    ret.finalError = f_norm;
    ret.numIterations = 1;
    ret.firstOrderOptimality = g_norm;
    if(_params.verbosity != VerbosityType::kSilent) {
      fprintf(stdout, "Converged. Initial value is optimal [%g < %g]\n", g_norm, _g_tol);
    }
    return ret;
  }

  _f_norm_prev = 0.0f;
  float dp_norm_prev = 0.0f;
  bool has_converged = false;

  while(ret.numIterations++ < _params.maxIterations && !has_converged &&
        _num_fun_evals < _params.maxFuncEvals) {

    if(!derived()->runIteration(tdata, channels, data, f_norm, ret.status)) {
      break;
    }

    data.T *= tdata->warp().paramsToPose(-data.dp); // update the pose

    float dp_norm = data.dp.norm();
    g_norm = data.gradientNorm();
    printIteration(ret.numIterations, f_norm, g_norm, dp_norm);

    has_converged = testConvergence(dp_norm, dp_norm_prev, g_norm, f_norm, ret.status);

    dp_norm_prev = dp_norm;
    _f_norm_prev = f_norm;
  }


  if(ret.status != PoseEstimationStatus::kSolverError)
    T = data.T;

  ret.numIterations -= 1; // because the ++
  ret.finalError = f_norm;
  ret.firstOrderOptimality = g_norm;

  if(_params.verbosity != VerbosityType::kSilent) {
    printResult(ret);
  }

  return ret;
}


}; // bpvo

#endif // BPVO_POSE_ESTIMATOR_BASE_H
