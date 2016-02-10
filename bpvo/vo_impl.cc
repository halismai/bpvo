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

#include "bpvo/vo_impl.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iostream>

namespace bpvo {

/**
 */
template <typename T> static inline
cv::Mat ToOpenCV(const T* data, int rows, int cols)
{
  return cv::Mat(rows, cols, cv::DataType<T>::type, (void*) data);
}

/**
 */
template <typename T> static inline
cv::Mat ToOpenCV(const T* data, const ImageSize& imsize)
{
  return ToOpenCV(data, imsize.rows, imsize.cols);
}

/**
 * \return number of pyramid levels such that tha coarsest level has size
 * min_allowed_res
 *
 * \param min_image_dim min(image.rows, image.cols)
 * \param min_allowed_res  minimum image resolution we want to work with
 */
static inline int getNumberOfPyramidLevels(int min_image_dim, int min_allowed_res)
{
  return 1 + std::round(std::log2(min_image_dim / (double) min_allowed_res));
}

static inline int getNumberOfPyramidLevels(const ImageSize& s, int min_allowed_res)
{
  return getNumberOfPyramidLevels(std::min(s.rows, s.cols), min_allowed_res);
}


VisualOdometry::Impl::Impl(const Matrix33& K, const float& b, ImageSize image_size,
                           AlgorithmParameters params)
    : _params(params)
    , _pose_est_params(params)
    , _pose_est_params_low_res(params)
    , _image_size(image_size)
    , _T_kf(Matrix44::Identity())
{
  assert( _image_size.numel() > 0 );

  //
  // set auto number of pyramid levels if needed
  //
  if(_params.numPyramidLevels <= 0)
    _params.numPyramidLevels = getNumberOfPyramidLevels(_image_size, 40);

  if(_params.relaxTolerancesForCoarseLevels)
    _pose_est_params_low_res.relaxTolerance();


  Matrix33 K_pyr(K);
  float b_pyr = b;
  _tdata_pyr.push_back(make_unique<TData>(K_pyr, b_pyr, 0));
  for(int i = 1; i < _params.numPyramidLevels; ++i) {
    K_pyr *= 0.5; K_pyr(2,2) = 1.0f;
    b_pyr *= 2.0;
    _tdata_pyr.push_back(make_unique<TData>(K_pyr, b_pyr, i));
  }

  for(int i= 0; i < _params.numPyramidLevels; ++i)
    _channels_pyr.push_back(ChannelsT(_params.sigmaPriorToCensusTransform,
                                      _params.sigmaBitPlanes));
}

VisualOdometry::Impl::~Impl() {}

Result VisualOdometry::Impl::addFrame(const uint8_t* I_ptr, const float* D_ptr)
{
  assert( !_channels_pyr.empty() && !_tdata_pyr.empty() );
  assert( _channels_pyr.size() == _tdata_pyr.size() );

  cv::Mat I;
  ToOpenCV(I_ptr, _image_size).copyTo(I);

  _channels_pyr.front().compute(I);
  for(size_t i = 1; i < _channels_pyr.size(); ++i) {
    cv::pyrDown(I, I);
    _channels_pyr[i].compute(I);
  }

  Result ret;
  ret.optimizerStatistics.resize(_channels_pyr.size());

  if(0 == _tdata_pyr.front()->numPoints()) {
    //
    // the first frame added to vo
    //
    cv::Mat D = ToOpenCV(D_ptr, _image_size);
    this->setAsKeyFrame(_channels_pyr, D);
    ret.pose = _T_kf;
    ret.covariance.setIdentity();
    ret.keyFramingReason = KeyFramingReason::kFirstFrame;
    ret.isKeyFrame = true;

    _kf_candidate.channels_pyr.resize(_channels_pyr.size()); // so we can swap later
    return ret;
  }

  THROW_ERROR_IF(_params.maxTestLevel != 0, "maxTestLevel must be 0 for now");

  //
  // initialize pose estimation for the next frame using the latest estimate
  //
  Matrix44 T_est = estimatePose(_channels_pyr, _T_kf, ret.optimizerStatistics);

  ret.keyFramingReason = shouldKeyFrame(T_est, _pose_estimator.getWeights());
  ret.isKeyFrame = ((int)ret.keyFramingReason != (int)KeyFramingReason::kNoKeyFraming);

  if(ret.isKeyFrame) {
    // if we do not have a keyframe candidate, this happens if keyframing is
    // disabled (e.g. minTranslationMagToKeyFrame = 0.0f), or if the first frame
    // motion was too large
    if(_kf_candidate.empty()) {
      setAsKeyFrame(_channels_pyr, ToOpenCV(D_ptr, _image_size));
      ret.pose = T_est * _T_kf.inverse();
      _T_kf.setIdentity();
    } else {
      //
      // set keyframe candidate as the current keyframe, and re-estimate the
      // pose with identity initialization
      //
      setAsKeyFrame(_kf_candidate);

      Matrix44 T_init = Matrix44::Identity();
      Matrix44 T_est = estimatePose(_channels_pyr, T_init, ret.optimizerStatistics);
      ret.pose = T_est;
      _T_kf = T_est;
      _kf_candidate.clear();
    }
  } else {
    // TODO test if there is enough good disparity estimates for the frame to be
    // assigned as a keyframe candidate
    _kf_candidate.channels_pyr.swap(_channels_pyr);
    _kf_candidate.disparity = ToOpenCV(D_ptr, _image_size).clone();
    ret.pose = T_est * _T_kf.inverse(); // return the relative motion wrt to the added frame
    _T_kf = T_est; // replace the initialization with the new motion
  }

  return ret;
}

Matrix44 VisualOdometry::
Impl::estimatePose(const std::vector<ChannelsT>& cn, const Matrix44& T_init,
                   std::vector<OptimizerStatistics>& stats)
{
  stats.resize(cn.size());
  Matrix44 T_est = T_init;

  THROW_ERROR_IF( _params.maxTestLevel != 0, "maxTestLevel must be 0 for now" );

  _pose_estimator.setParameters(_pose_est_params_low_res);
  int i = static_cast<int>(cn.size()) - 1;
  for( ; i >= _params.maxTestLevel;  --i) {
    dprintf("level %d/%d\n", i, _params.maxTestLevel);
    if(i == 0) { // set the original thresholds for the finest pyramid level
      _pose_estimator.setParameters(_pose_est_params);
    }
    stats[i] = _pose_estimator.run(_tdata_pyr[i].get(), cn[i], T_est);
  }

  return T_est;
}

void VisualOdometry::Impl::setAsKeyFrame(const std::vector<ChannelsT>& cn,
                                         const cv::Mat& disparity)
{
  assert( _tdata_pyr.size() == cn.size() );

  for(size_t i = 0; i < cn.size(); ++i)
    _tdata_pyr[i]->setData(cn[i], disparity);
}

int VisualOdometry::Impl::numPointsAtLevel(int level) const
{
  assert( level >= 0 && level < (int) _tdata_pyr.size() );

  return _tdata_pyr[level]->numPoints();
}

auto VisualOdometry::Impl::pointsAtLevel(int level) const -> const PointVector&
{
  assert( level >= 0 && level < (int) _tdata_pyr.size() );

  return _tdata_pyr[level]->points();
}

KeyFramingReason
VisualOdometry::Impl::shouldKeyFrame(const Matrix44& T, const std::vector<float>& weights)
{
  auto t_norm = T.block<3,1>(0,3).squaredNorm();
  if(t_norm > math::sq(_params.minTranslationMagToKeyFrame)) {
    return KeyFramingReason::kLargeTranslation;
  }

  auto r_norm = math::RotationMatrixToEulerAngles(T).squaredNorm();
  if(r_norm > math::sq(_params.minRotationMagToKeyFrame)) {
    return KeyFramingReason::kLargeRotation;
  }

  auto thresh = _params.goodPointThreshold;
  auto num_good = std::count_if(std::begin(weights), std::end(weights),
                                [=](float w) { return w > thresh; });
  float frac_good = num_good / (float) weights.size();
  dprintf("fraction of good points %0.2f at threshold %f\n", frac_good, thresh);
  if(frac_good < _params.maxFractionOfGoodPointsToKeyFrame) {
    return KeyFramingReason::kSmallFracOfGoodPoints;
  }

  return KeyFramingReason::kNoKeyFraming;
}

bool VisualOdometry::Impl::KeyFrameCandidate::empty() const
{
  return disparity.empty();
}

void VisualOdometry::Impl::KeyFrameCandidate::clear()
{
  disparity = cv::Mat();
}

} // bpvo

