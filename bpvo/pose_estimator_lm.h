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

#ifndef BPVO_POSE_ESTIMATOR_LM_H
#define BPVO_POSE_ESTIMATOR_LM_H

#include <bpvo/pose_estimator_base.h>
#include <bpvo/debug.h>
#include <bpvo/linear_system_builder.h>
#include <Eigen/Cholesky>

namespace bpvo {

template <class TDataT>
class PoseEstimatorLM : public PoseEstimatorBase< PoseEstimatorLM<TDataT> >
{
 public:
  typedef PoseEstimatorLM<TDataT> Self;
  typedef PoseEstimatorBase<Self> Base;

  using typename Base::Warp;
  using typename Base::Jacobian;
  using typename Base::Gradient;
  using typename Base::ParameterVector;
  using typename Base::Hessian;
  using typename Base::TemplateData;
  using typename Base::PoseEstimatorData;

 public:
  PoseEstimatorLM(const PoseEstimatorParameters& params = PoseEstimatorParameters())
  {
    Base::setParameters(params);
  }

  inline std::string name() const { return "LevenbergMarquardt"; }

  inline float linearize(const TemplateData* tdata, const DenseDescriptor* channels,
                         PoseEstimatorData& data, bool with_hessian = true)
  {
    tdata->computeResiduals(channels, data.T, Base::residuals(), Base::valid());
    this->replicateValidFlags();
    auto sigma = this->_scale_estimator.estimateScale(Base::residuals(), Base::valid());
    MEstimator::ComputeWeights(this->_params.lossFunction, Base::residuals(),
                               Base::valid(), sigma, Base::weights());

    this->_num_fun_evals += 1;

    auto* H = with_hessian ? &data.H : NULL;
    auto* G = with_hessian ? &data.G : NULL;
    return LinearSystemBuilder::Run(tdata->jacobians(), Base::residuals(),
                                    Base::weights(), Base::valid(), H, G);
  }

  inline bool runIteration(const TemplateData* tdata, const DenseDescriptor* channels,
                           PoseEstimatorData& data, float& f_norm, PoseEstimationStatus& status)
  {
    bool do_accept_step = false;
    float dp_norm_prev = data.dp.norm();
    float eps2 = this->_params.parameterTolerance;

    f_norm = this->linearize(tdata, channels, data, true);

    do {
      printf("u: %f\n", _u);
      if(!data.solve2Augmented(_u)) {
        if(this->_params.verbosity != VerbosityType::kSilent) {
          Warn("Solver failed\n");
        }
        status = PoseEstimationStatus::kSolverError;
        return false;
      }

      float dp_norm = data.dp.norm();
      if(dp_norm <= eps2*(dp_norm_prev + eps2)) {
        do_accept_step = true;
        status = PoseEstimationStatus::kParameterTolReached;
      }
      dp_norm_prev = dp_norm;

      float f_new = this->linearize(tdata, channels, data, false);
      auto dl = 0.5f * data.dp.transpose() * (_u * data.dp - data.G);
      float rho = (f_norm - f_new) / dl;

      if(rho > 0) { // accept the step
        f_norm = f_new;
        float r= 2.0f*rho - 1;
        _u = _u * std::max(1.0f / 3.0f, 1 - r*r*r);
        _v = 2.0f;
        do_accept_step = true;
        printf("ACCEPT\n");
        f_new = this->linearize(tdata, channels, data, true);
      } else {
        _u = _u * _v;
        _v = 2.0f * _v;
        printf("NOT %f %f\n", _u, _v);
      }

    } while(!do_accept_step && this->_num_fun_evals < this->_params.maxFuncEvals);

    return do_accept_step;
  }

 private:
  float _u = 0.0f, _v = 2.0f;
}; // PoseEstimatorLM


}; // bpvo

#endif // BPVO_POSE_ESTIMATOR_LM_H
