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

#ifndef BPVO_VO_POSE_ESTIMATOR_H
#define BPVO_VO_POSE_ESTIMATOR_H

#include <bpvo/pose_estimator_gn.h>
#include <bpvo/template_data.h>
#include <bpvo/types.h>

namespace bpvo {

class VisualOdometryFrame;
class OptimizerLM;

class VisualOdometryPoseEstimator
{
 public:
  VisualOdometryPoseEstimator(const AlgorithmParameters&);
  ~VisualOdometryPoseEstimator();

  /**
   * Estimate the pose of the cur_frame wrt to ref_frame
   *
   * \return optimizerStatistics per pyrmaid level
   *
   * \param T_init pose initialization
   * \param T_est  estimated pose
   */
  std::vector<OptimizerStatistics>
  estimatePose(const VisualOdometryFrame* ref_frame,
               const VisualOdometryFrame* cur_frame,
               const Matrix44& T_init,
               Matrix44& T_est);

  float getFractionOfGoodPoints(float thresh) const;

  const WeightsVector& getWeights() const;

 private:
  AlgorithmParameters _params;
  PoseEstimatorGN<TemplateData> _pose_estimator;
  PoseEstimatorParameters _pose_est_params;
  PoseEstimatorParameters _pose_est_params_low_res;
  //UniquePointer<OptimizerLM> _optimizer;
}; // VisualOdometryPoseEstimator

}; // bpvo

#endif // BPVO_VO_POSE_ESTIMATOR_H

