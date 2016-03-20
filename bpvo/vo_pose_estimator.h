#ifndef BPVO_VO_POSE_ESTIMATOR_H
#define BPVO_VO_POSE_ESTIMATOR_H

#include <bpvo/pose_estimator_gn.h>
#include <bpvo/template_data.h>
#include <bpvo/types.h>

namespace bpvo {

class VisualOdometryFrame;

class VisualOdometryPoseEstimator
{
 public:
  VisualOdometryPoseEstimator(const AlgorithmParameters&);

  /**
   * Estimate the pose of the cur_frame wrt to ref_frame
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

 private:
  AlgorithmParameters _params;
  PoseEstimatorGN<TemplateData> _pose_estimator;
  PoseEstimatorParameters _pose_est_params;
  PoseEstimatorParameters _pose_est_params_low_res;
}; // VisualOdometryPoseEstimator

}; // bpvo

#endif // BPVO_VO_POSE_ESTIMATOR_H
