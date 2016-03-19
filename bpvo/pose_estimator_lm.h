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

#include <bpvo/pose_estimator_gn.h>

namespace bpvo {

template <typename TDataT>
class PoseEstimatorLM : public PoseEstimatorBase< PoseEstimatorLM<TDataT> >
{
 public:
  typedef PoseEstimatorGN<TDataT> Self;
  typedef PoseEstimatorBase<Self> Base;

  using typename Base::Warp;
  using typename Base::Jacobian;
  using typename Base::Gradient;
  using typename Base::ParameterVector;
  using typename Base::Hessian;
  using typename Base::TemplateData;
  using typename Base::PoseEstimatorData;

 public:
  /**
   * \param params pose estimation parameters
   */
  PoseEstimatorLM(const PoseEstimatorParameters& params = PoseEstimatorParameters())
  {
    Base::setParameters(params);
  }

  /**
   * \return the )ame of the algorithm
   */
  inline std::string name() const { return "PoseEstimatorLM"; }


  /**
   * \param tdata       template data
   * \param channels    input channels
   * \param pose        current pose
   * \param H           hessian [optional]
   * \param G           gradien [optional]
   * \return            functionv value (norm of the residuals vector)
   */
  inline float linearize(TemplateData* tdata, const DenseDescriptor* channels, PoseEstimatorData& data)
  {
    tdata->computeResiduals(channels, data.T, Base::residuals(), Base::valid());
    this->replicateValidFlags();
    auto sigma = this->_scale_estimator.estimateScale(Base::residuals(), Base::valid());
    MEstimator::ComputeWeights(this->_params.lossFunction, Base::residuals(),
                               Base::valid(), sigma, Base::weights());

    this->_num_fun_evals += 1;
    return LinearSystemBuilder::Run(
        tdata->jacobians(), Base::residuals(), Base::weights(), Base::valid(), &data.H, &data.G);
  }

  inline bool runIteration(TemplateData* tdata, const DenseDescriptor* channels,
                           PoseEstimatorData& data, float& f_norm,
                           PoseEstimationStatus& status)

  {
    f_norm = this->linearize(tdata, channels, data);

    if(!data.solve()) {
      if(this->_params.verbosity != VerbosityType::kSilent)
        Warn("solver failed");
      status = PoseEstimationStatus::kSolverError;
      return false;
    }

    return true;
  }

  inline void reset()
  {
    Base::reset();
    _initial_lambda = 1e-4;
    _v = 2.0f;
    _u = 1.0f;
  }

 private:
  float _initial_lambda = 1e-4;
  float _v = 2.0f, _u = 1.0f;
}; // PoseEstimatorLM

}; // bpvo

#endif // BPVO_POSE_ESTIMATOR_LM_H

