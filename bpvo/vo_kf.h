#ifndef BPVO_VO_KF_H
#define BPVO_VO_KF_H

#include <bpvo/types.h>
#include <bpvo/pose_estimator_gn.h>
#include <bpvo/template_data.h>

namespace cv { class Mat; }; // cv

namespace bpvo {

class VisualOdometryPoseEstimator;
class DenseDescriptorPyramid;

class VisualOdometryWithKeyFraming
{
 public:
  /**
   */
  VisualOdometryWithKeyFraming(const Matrix33& K, const float baseline,
                               ImageSize, AlgorithmParameters);

  ~VisualOdometryWithKeyFraming();

  /**
   */
  Result addFrame(const uint8_t* image_ptr, const float* disparity_ptr);

  int numPointsAtLevel(int level = -1) const;

 private:
  AlgorithmParameters _params;
  ImageSize _image_size;
  UniquePointer<VisualOdometryPoseEstimator> _vo_pose;
  Matrix44 _T_kf;

  struct KeyFrameCandidate;
  UniquePointer<KeyFrameCandidate> _kf_candidate;

  UniquePointer<DenseDescriptorPyramid> _desc_pyr;

 private:
  KeyFramingReason shouldKeyFrame(const Result&);
}; // VisualOdometryWithKeyFraming

}; // bpvo

#endif // BPVO_VO_KF_H
