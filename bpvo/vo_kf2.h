#ifndef BPVO_VO_KF2_H
#define BPVO_VO_KF2_H

#include <bpvo/types.h>

namespace bpvo {

class VisualOdometryFrame;
class VisualOdometryPoseEstimator;

class VisualOdometryKeyFraming
{
  typedef typename EigenAlignedContainer<Matrix44>::type PoseList;

 public:
  VisualOdometryKeyFraming(const Matrix33& K, float b, ImageSize, AlgorithmParameters);
  ~VisualOdometryKeyFraming();

  Result addFrame(const uint8_t*, const float*);

  inline const PoseList& getKeyFramePoses() const { return _kf_poses; }

 private:
  AlgorithmParameters _params;
  ImageSize _image_size;
  UniquePointer<VisualOdometryPoseEstimator> _vo_pose;
  UniquePointer<VisualOdometryFrame> _ref_frame;
  UniquePointer<VisualOdometryFrame> _cur_frame;
  UniquePointer<VisualOdometryFrame> _prev_frame;
  Matrix44 _T_kf;

  PoseList _kf_poses;

  KeyFramingReason shouldKeyFrame(const Matrix44&) const;
}; // VisualOdometryKeyFraming


}; // bpvo

#endif
