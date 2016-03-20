#include "bpvo/vo_no_kf.h"
#include "bpvo/vo_frame.h"
#include "bpvo/vo_pose_estimator.h"
#include "bpvo/opencv.h"

#include <cmath>
#include <condition_variable>

namespace bpvo {

VisualOdometryNoKeyFraming::
VisualOdometryNoKeyFraming(
const Matrix33& K, float b, ImageSize s, AlgorithmParameters p)
: _image_size(s), _vo_pose(make_unique<VisualOdometryPoseEstimator>(p))
{
  if(p.numPyramidLevels <= 0)
    p.numPyramidLevels = 1 + std::round(
        std::log2(std::min(s.rows, s.cols) / (double) p.minImageDimensionForPyramid));

  _ref_frame = make_unique<VisualOdometryFrame>(K, b, p);
  _cur_frame = make_unique<VisualOdometryFrame>(K, b, p);
}

VisualOdometryNoKeyFraming::~VisualOdometryNoKeyFraming() {}

Result FirstFrameResult(int n_levels)
{
  Result r;
  r.pose.setIdentity();
  r.covariance.setIdentity();
  r.optimizerStatistics.resize(n_levels);
  r.isKeyFrame = true;
  r.keyFramingReason = KeyFramingReason::kFirstFrame;

  return r;
}

Result VisualOdometryNoKeyFraming::
addFrame(const uint8_t* image_ptr, const float* disparity_ptr)
{
  const auto I = ToOpenCV(image_ptr, _image_size);
  const auto D = ToOpenCV(disparity_ptr, _image_size);

  _cur_frame->setData(I, D);

  Result ret;
  if(!_ref_frame->empty())
  {
    std::unique_lock<std::mutex> lock(_ref_frame->mutex());
    std::condition_variable cond;
    cond.wait(lock, [&]() { return _ref_frame->isTemplateDataReady(); });

    ret.optimizerStatistics = _vo_pose->estimatePose(
        _ref_frame.get(), _cur_frame.get(), Matrix44::Identity(), ret.pose);
    ret.isKeyFrame = true;
    ret.keyFramingReason = KeyFramingReason::kLargeTranslation;
  } else
  {
    ret = FirstFrameResult(_ref_frame->numLevels());
  }

  std::swap(_ref_frame, _cur_frame);
  return ret;
}


} // bpvo


