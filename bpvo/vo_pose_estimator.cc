/*
   This file is part of bpvo.

   bpvo is free software: you can redistribute it and/or modify
   it under the terms of the Lesser GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   bpvo is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   Lesser GNU General Public License for more details.

   You should have received a copy of the Lesser GNU General Public License
   along with bpvo.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Contributor: halismai@cs.cmu.edu
 */

#include <bpvo/vo_pose_estimator.h>
#include <bpvo/vo_frame.h>

#include <algorithm>

namespace bpvo {

class OptimizerLM
{
 public:
  OptimizerLM(const PoseEstimatorParameters& params = PoseEstimatorParameters())
      : _params(params), _u(0.0f), _v(2.0f) {}

  void setParameters(const PoseEstimatorParameters& p) { _params = p; }

  OptimizerStatistics run(const TemplateData* tdata, const DenseDescriptor*,
                          Matrix44&);

  inline const WeightsVector& getWeights() const
  {
    printf("getWeights %zu\n", _weights.size());
    return _weights;
  }

 private:
  PoseEstimatorParameters _params;
  float _u = 0.0f, _v = 2.0f;

  ResidualsVector _residuals;
  WeightsVector _weights;
  ValidVector _valid;
}; // OptimizerLM

VisualOdometryPoseEstimator::VisualOdometryPoseEstimator(const AlgorithmParameters& p)
    : _params(p)
    , _pose_est_params(p)
    , _pose_est_params_low_res(p) {}
    //, _optimizer(new OptimizerLM(_pose_est_params)) {}

VisualOdometryPoseEstimator::~VisualOdometryPoseEstimator() {}

std::vector<OptimizerStatistics>
VisualOdometryPoseEstimator::estimatePose(
    const VisualOdometryFrame* ref_frame, const VisualOdometryFrame* cur_frame,
    const Matrix44& T_init, Matrix44& T_est)
{
  std::vector<OptimizerStatistics> ret(ref_frame->numLevels());

  T_est = T_init;
  _pose_estimator.setParameters(_pose_est_params_low_res);

  /*OptimizerLM optimizer;
  optimizer.setParameters(_pose_est_params_low_res);*/

  for(int i = ref_frame->numLevels()-1; i >= _params.maxTestLevel; --i)
  {
    if(i >= _params.maxTestLevel) {
      _pose_estimator.setParameters(_pose_est_params);
      //optimizer.setParameters(_pose_est_params);
    }

    ret[i] = _pose_estimator.run(ref_frame->getTemplateDataAtLevel(i),
                                 cur_frame->getDenseDescriptorAtLevel(i),
                                 T_est);
    /*
    ret[i] = optimizer.run(ref_frame->getTemplateDataAtLevel(i),
                           cur_frame->getDenseDescriptorAtLevel(i),
                           T_est);*/
  }

  return ret;
}

const WeightsVector& VisualOdometryPoseEstimator::getWeights() const
{
  return _pose_estimator.getWeights();
  //return _optimizer->getWeights();
}

float VisualOdometryPoseEstimator::getFractionOfGoodPoints(float thresh) const
{
  const auto& w = _pose_estimator.getWeights();
  //const auto& w = _optimizer->getWeights();
  auto n = std::count_if(w.begin(), w.end(), [=](float w_i) { return w_i > thresh; });
  return n / static_cast<float>(w.size());
}

inline OptimizerStatistics
OptimizerLM::run(const TemplateData* tdata, const DenseDescriptor* channels,
                 Matrix44& T_est)
{
  OptimizerStatistics ret;

  float sigma = 1.0f;
  AutoScaleEstimator scale_estimator;

  Eigen::Matrix<float,6,6> H;
  Eigen::Matrix<float,6,1> G;
  Eigen::Matrix<float,6,1> dp;

  int num_func_evals = 0;

  auto Solve = [&]()
  {
    decltype(H) H_a(H);
    H_a.diagonal().array() += _u;
    dp = H_a.ldlt().solve(G);
    bool ret =  (H_a * dp).isApprox(G);
    if(!ret) {
      std::cout << H_a << std::endl;
      std::cout << "u:" << _u << std::endl;
      std::cout << "G:" << G.transpose() << std::endl;
      std::cout << "p:" << dp.transpose() << std::endl;
    }
    return ret;
  }; // Solve

  /**
   * replicate the valid flags so that they match the number of residuals
   */
  auto ReplicateValidFlags = [&]()
  {
    if(_residuals.size() != _valid.size()) {
      int m = _residuals.size() / _valid.size();
      decltype(_valid) tmp(_residuals.size());
      auto* ptr = tmp.data();
      for(int i = 0; i < m; ++i, ptr += _valid.size()) {
        memcpy(ptr, _valid.data(), _valid.size() * sizeof(_valid[0]));
      }
    }
  };

  /**
   * re-compute the weights for IRLS
   */
  auto ComputeWeights = [&]()
  {
    sigma = scale_estimator.estimateScale(_residuals, _valid);
    MEstimator::ComputeWeights(_params.lossFunction, _residuals, _valid,
                               sigma, _weights);
  }; // ComputeWeights


  /**
   * Evaluate the objective, computing residuals and valid flags
   */
  auto EvalFunc = [&](const Matrix44& pose_)
  {
    tdata->computeResiduals(channels, pose_, _residuals, _valid);
    ReplicateValidFlags();
    ComputeWeights();

    num_func_evals++;
  }; // EvalFunc


  /**
   * If with_hessian = true, we re-comptue the Hessian and gradient
   *
   * \return the objective norm
   */
  auto Linearize = [&](bool with_hessian_ = true)
  {
    return LinearSystemBuilder::Run(tdata->jacobians(), _residuals, _weights, _valid,
                                    with_hessian_ ? &H : nullptr, with_hessian_ ? &G : nullptr);
  }; // Linearize

  static constexpr const char* _verbose_fmt_str_first_it =
      " %5d       %5d   %13.6g    %12.3g\n";
  static constexpr const char* _verbose_fmt_str =
      " %5d       %5d   %13.6g    %12.3g    %12.6g    %12.6g\n";

  _u = 0.0f; _v = 2.0f;
  auto p_thresh = _params.parameterTolerance;
  auto f_thresh = _params.functionTolerance;

  //
  // compute the first iteration outside
  //
  EvalFunc(T_est);
  auto f_norm = Linearize(true);
  auto g_norm = G.lpNorm<Eigen::Infinity>();

  printf("\n                                        First-Order         Norm of       Delta\n"
             " Iteration  Func-count    Residual       optimality            step       error\n");
  fprintf(stdout, _verbose_fmt_str_first_it, 0, num_func_evals, f_norm, g_norm);

  if(g_norm < _params.gradientTolerance) {
    Info("initial point is optimal\n");
    ret.numIterations = 1;
    ret.firstOrderOptimality = g_norm;
    ret.status = PoseEstimationStatus::kGradientTolReached;
    ret.finalError = f_norm;
    return ret;
  }

  if(f_norm < f_thresh) {
    Info("function value is too small\n");
  }

  bool found = false;
  _u = 1e-6 * ( H.diagonal().maxCoeff() / (double) tdata->numPoints());
  _u = 1e-6;

  ret.status = PoseEstimationStatus::kMaxIterations;
  Matrix44 T_new(Matrix44::Identity());
  float dp_norm_prev = 0.0f; //< TODO get it from T_est
  int it = 1;
  while(!found && it < _params.maxIterations && num_func_evals < _params.maxFuncEvals) {
    printf("u: %f\n", _u);
    if(!Solve()) {
      Fatal("bad\n");
    }
    auto dp_norm = dp.norm();
    if(dp_norm <= p_thresh * (dp_norm_prev + p_thresh)) {
      found = true;
      ret.status = PoseEstimationStatus::kParameterTolReached;
      Info("got solution %f\n", dp_norm);
    } else {
      Matrix44 T_new = tdata->warp().paramsToPose(-dp) * T_est;
      EvalFunc(T_new);
      auto f_norm_new = Linearize(false);
      auto rho = (f_norm - f_norm_new) / (0.5f*dp.transpose()*(_u*dp - G));
      if(rho > 0) {
        T_est = T_new;
        f_norm = Linearize(true);
        found = G.lpNorm<Eigen::Infinity>() <= _params.gradientTolerance;
        auto r = 2*rho - 1;
        _u *= std::max(1.0f/3.0f, 1.0f - r*r*r);
        _v = 2.0f;
        dp_norm_prev = dp_norm;

        ++it;
        fprintf(stdout, _verbose_fmt_str, it, num_func_evals, f_norm, g_norm, dp_norm,
                f_norm_new - f_norm);
      } else {
        _u *= _v;
        _v *= 2.0f;
      }
    }
  }

  printf("weights %zu points %zu\n", _weights.size(), tdata->points().size());

  ret.numIterations = it-1;
  ret.finalError = f_norm;
  ret.firstOrderOptimality = g_norm;

  return ret;
}

}; // bpvo

