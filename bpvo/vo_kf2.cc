#include "bpvo/vo_kf2.h"
#include "bpvo/vo_frame.h"
#include "bpvo/vo_pose_estimator.h"
#include "bpvo/opencv.h"

namespace bpvo {


VisualOdometryKeyFraming::
VisualOdometryKeyFraming(const Matrix33& K, float b, ImageSize s, AlgorithmParameters p)
 : _params(p), _image_size(s), _vo_pose(make_unique<VisualOdometryPoseEstimator>(p)), _T_kf(Matrix44::Identity())
{
  if(_params.numPyramidLevels <= 0)
    _params.numPyramidLevels = 1 + std::round(
        std::log2(std::min(s.rows, s.cols) / (double) p.minImageDimensionForPyramid));

  _ref_frame = make_unique<VisualOdometryFrame>(K, b, _params);
  _cur_frame = make_unique<VisualOdometryFrame>(K, b, _params);
  _prev_frame = make_unique<VisualOdometryFrame>(K, b, _params);

}

VisualOdometryKeyFraming::~VisualOdometryKeyFraming() {}

static inline Result FirstFrameResult(int n_levels)
{
  Result r;
  r.pose.setIdentity();
  r.covariance.setIdentity();
  r.optimizerStatistics.resize(n_levels);
  r.isKeyFrame = true;
  r.keyFramingReason = KeyFramingReason::kFirstFrame;

  return r;
}

Result VisualOdometryKeyFraming::
addFrame(const uint8_t* I_ptr, const float* D_ptr)
{
  const auto I = ToOpenCV(I_ptr, _image_size);
  const auto D = ToOpenCV(D_ptr, _image_size);

  _cur_frame->setData(I, D);

  if(!_ref_frame->hasTemplate()) {
    std::swap(_ref_frame, _cur_frame);
    _ref_frame->setTemplate();
    _kf_poses.push_back( _T_kf );
    return FirstFrameResult(_ref_frame->numLevels());
  }

  Matrix44 T_est;

  Result ret;
  ret.optimizerStatistics = _vo_pose->estimatePose(_ref_frame.get(), _cur_frame.get(),
                                                   _T_kf, T_est);
  ret.keyFramingReason = shouldKeyFrame(T_est);
  ret.isKeyFrame = KeyFramingReason::kNoKeyFraming != ret.keyFramingReason;

  if(!ret.isKeyFrame)
  {
    // store _cur_frame in _prev_frame as a keyframe candidate for the future
    std::swap(_prev_frame, _cur_frame);
    ret.pose = T_est * _T_kf.inverse(); // return the relative motion
    _T_kf = T_est; // accumulate the intitialization
  } else
  {
    if(_prev_frame->empty())
    {
      //
      // we were unable to obtain an intermediate frame (either because
      // keyframing is turned off, or all estimated motions thus far satisfy the
      // keyframine criteria
      //
      std::swap(_cur_frame, _ref_frame);
      _ref_frame->setTemplate();

      _kf_poses.push_back( T_est.inverse() );

      ret.pose = T_est *  _T_kf.inverse();
      _T_kf.setIdentity(); // reset intiialization
    } else
    {
      _kf_poses.push_back( _T_kf );

      std::swap(_prev_frame, _ref_frame);
      _prev_frame->clear(); // no longer a suitable candidate for keyframing
      _ref_frame->setTemplate();

      //
      // re-estimate the motion, because the estimate than caused keyframing is
      // most likely bogus
      //
      Matrix44 T_init(Matrix44::Identity());
      ret.optimizerStatistics = _vo_pose->estimatePose(_ref_frame.get(), _cur_frame.get(),
                                                       T_init, T_est);
      ret.pose = T_est;
      _T_kf = T_est;
    }
  }

  return ret;
}


KeyFramingReason VisualOdometryKeyFraming::shouldKeyFrame(const Matrix44& pose) const
{
  auto t_norm = pose.block<3,1>(0,3).squaredNorm();
  if(t_norm > math::sq(_params.minTranslationMagToKeyFrame))
  {
    dprintf("keyFramingReason::kLargeTranslation\n");
    return KeyFramingReason::kLargeTranslation;
  }

  auto r_norm = math::RotationMatrixToEulerAngles(pose).squaredNorm();
  if(r_norm > math::sq(_params.minRotationMagToKeyFrame))
  {
    dprintf("kLargeRotation\n");
    return KeyFramingReason::kLargeRotation;
  }

  auto frac_good = _vo_pose->getFractionOfGoodPoints(_params.goodPointThreshold);
  if(frac_good < _params.maxFractionOfGoodPointsToKeyFrame)
  {
    dprintf("kSmallFracOfGoodPoints\n");
    return KeyFramingReason::kSmallFracOfGoodPoints;
  }

  return KeyFramingReason::kNoKeyFraming;
}

}; // bpvo


