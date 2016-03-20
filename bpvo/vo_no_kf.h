#ifndef BPVO_VO_NO_KF_H
#define BPVO_VO_NO_KF_H

#include <bpvo/types.h>

namespace bpvo {

class VisualOdometryFrame;
class VisualOdometryPoseEstimator;

class VisualOdometryNoKeyFraming
{
 public:
  VisualOdometryNoKeyFraming(const Matrix33& K, float b, ImageSize, AlgorithmParameters);
  ~VisualOdometryNoKeyFraming();

  Result addFrame(const uint8_t*, const float*);

 private:
  ImageSize _image_size;
  UniquePointer<VisualOdometryPoseEstimator> _vo_pose;
  UniquePointer<VisualOdometryFrame> _ref_frame;
  UniquePointer<VisualOdometryFrame> _cur_frame;
}; // VisualOdometryNoKeyFraming

}; // bpvo

#endif // BPVO_VO_NO_KF_H
