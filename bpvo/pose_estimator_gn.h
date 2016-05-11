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

#ifndef BPVO_POSE_ESTIMATOR_GN_H
#define BPVO_POSE_ESTIMATOR_GN_H

#include <bpvo/pose_estimator_base.h>
#include <bpvo/debug.h>
#include <bpvo/linear_system_builder.h>
#include <Eigen/Cholesky>

namespace bpvo {

template <class TDataT>
class PoseEstimatorGN : public PoseEstimatorBase< PoseEstimatorGN<TDataT> >
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
  PoseEstimatorGN(const PoseEstimatorParameters& params = PoseEstimatorParameters())
  {
    Base::setParameters(params);
  }

  /**
   * \return the name of the algorithm
   */
  inline std::string name() const { return "PoseEstimatorGN"; }


  /**
   * \param tdata       template data
   * \param channels    input channels
   * \param pose        current pose
   * \param H           hessian [optional]
   * \param G           gradien [optional]
   * \return            functionv value (norm of the residuals vector)
   */
  inline float linearize(const TemplateData* tdata, const DenseDescriptor* channels, PoseEstimatorData& data)
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

  inline bool runIteration(const TemplateData* tdata, const DenseDescriptor* channels,
                           PoseEstimatorData& data, float& f_norm,
                           PoseEstimationStatus& status)

  {
    f_norm = this->linearize(tdata, channels, data);

    if(!data.solve()) {
      if(this->_params.verbosity != VerbosityType::kSilent)
        Warn("solver failed");

      Warn("Solver failed\n");
      status = PoseEstimationStatus::kSolverError;
      return false;
    }

    return true;
  }
}; // PoseEstimatorGN

}; // bpvo

#endif // BPVO_POSE_ESTIMATOR_GN_H
