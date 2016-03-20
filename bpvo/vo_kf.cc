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

#include "bpvo/vo_kf.h"
#include "bpvo/utils.h"
#include "bpvo/parallel_tasks.h"
#include "bpvo/dense_descriptor_pyramid.h"
#include "bpvo/opencv.h"

#include <cmath>

#include <opencv2/highgui/highgui.hpp>

namespace bpvo {

class VisualOdometryPoseEstimator
{
  friend class VisualOdometryWithKeyFraming;

 public:
  /**
   * \param K the intrinsics matrix
   * \param b the stereo baseline
   * \param AlgorithmParameters parameters
   */
  VisualOdometryPoseEstimator(const Matrix33& K, const float baseline, AlgorithmParameters);

  /**
   * \param DenseDescriptorPyramid
   * \param D the disparity at the finest resolution
   */
  VisualOdometryPoseEstimator& setTemplate(const DenseDescriptorPyramid&, const cv::Mat& D);

  /**
   * Estimates the pose wrt to the TemplateData
   *
   * \param DenseDescriptorPyramid of the input frame data
   * \param T_init pose initialization
   * \param T_est  the estimated pose
   *
   * NOTE: setTemplate must be called before calling this function
   */
  std::vector<OptimizerStatistics>
  estimatePose(DenseDescriptorPyramid&, const Matrix44& T_init, Matrix44& T_est);

  /**
   * \return true if template data was set
   */
  bool hasTemplate() const;

 private:
  AlgorithmParameters _params;
  PoseEstimatorGN<TemplateData> _pose_estimator;
  PoseEstimatorParameters _pose_est_params;
  PoseEstimatorParameters _pose_est_params_low_res;
  std::vector<UniquePointer<TemplateData>> _tdata_pyr;

 protected:
  OptimizerStatistics estimatePosetAtLevel(int level, DenseDescriptorPyramid&,
                                           const Matrix44& T_init, Matrix44& T_est);
}; // VisualOdometryPoseEstimator


VisualOdometryPoseEstimator::
VisualOdometryPoseEstimator(const Matrix33& K, const float b, AlgorithmParameters p)
  : _params(p)
  , _pose_estimator()
  , _pose_est_params(p)
  , _pose_est_params_low_res(p)
  , _tdata_pyr(p.numPyramidLevels)
{

  THROW_ERROR_IF(_tdata_pyr.empty(), "numPyramidLevels must be > 0");

  Matrix33 K_pyr(K);
  float b_pyr(b);
  _tdata_pyr.front() = make_unique<TemplateData>(0, K_pyr, b_pyr, p);
  for(size_t i = 1; i < _tdata_pyr.size(); ++i)
  {
    K_pyr *= 0.5; K_pyr(2,2) = 1.0; b_pyr *= 2.0;
    _tdata_pyr[i] = make_unique<TemplateData>(i, K_pyr, b_pyr, p);
  }
}

VisualOdometryPoseEstimator& VisualOdometryPoseEstimator::
setTemplate(const DenseDescriptorPyramid& desc_pyr, const cv::Mat& D)
{
  // TODO no need for parallel stuff here, too much overhead
#if 0
  ParallelTasks tasks(std::min(desc_pyr.size(), 4));

  for(int i = 0; i < desc_pyr.size(); ++i)
  {
    tasks.add([=,&desc_pyr]() {
          desc_pyr.compute(i);
          _tdata_pyr[i]->setData(desc_pyr[i], D);
        });
  }

  tasks.wait();
#endif

  for(int i = desc_pyr.size()-1; i >= _params.maxTestLevel; --i)
    _tdata_pyr[i]->setData(desc_pyr[i], D);

  /*
  ParallelTasks tasks(std::min(desc_pyr.size(), 4));
  for(int i = desc_pyr.size()-1; i >= _params.maxTestLevel; --i)
    tasks.add([=,&desc_pyr]() { _tdata_pyr[i]->setData(desc_pyr[i], D); });
  tasks.wait();
  */

  return *this;
}

std::vector<OptimizerStatistics> VisualOdometryPoseEstimator::
estimatePose(DenseDescriptorPyramid& desc_pyr, const Matrix44& T_init, Matrix44& T_est)
{
  std::vector<OptimizerStatistics> ret(desc_pyr.size());
  _pose_estimator.setParameters(_pose_est_params_low_res);
  T_est = T_init;
  for(int i = desc_pyr.size()-1; i >= _params.maxTestLevel; --i)
  {
    ret[i] = estimatePosetAtLevel(i, desc_pyr, T_est, T_est);
  }

  return ret;
}

OptimizerStatistics VisualOdometryPoseEstimator::
estimatePosetAtLevel(int level, DenseDescriptorPyramid& desc_pyr,
                     const Matrix44& T_init, Matrix44& T_est)
{
  if(level >= _params.maxTestLevel)
    _pose_estimator.setParameters(_pose_est_params);

  //desc_pyr.compute(level);
  T_est = T_init;
  return _pose_estimator.run(_tdata_pyr[level].get(), desc_pyr[level], T_est);
}

bool VisualOdometryPoseEstimator::hasTemplate() const
{
  return _tdata_pyr[_params.maxTestLevel]->numPoints() > 0;
}

struct VisualOdometryWithKeyFraming::KeyFrameCandidate
{
  KeyFrameCandidate(const AlgorithmParameters& p)
      : _has_data(false), _desc_pyr(p), _disparity() {}

  void set(const DenseDescriptorPyramid& desc_pyr, const cv::Mat& D)
  {
    D.copyTo(_disparity);
    desc_pyr.copyTo(_desc_pyr);
    _has_data = true;
  }

  bool empty() const { return !_has_data; }
  void clear() { _has_data = false; }

  bool _has_data = false;
  DenseDescriptorPyramid _desc_pyr;
  cv::Mat _disparity;
}; // KeyFrameCandidate

VisualOdometryWithKeyFraming::
VisualOdometryWithKeyFraming(const Matrix33& K, const float b,
                             ImageSize s, AlgorithmParameters p)
  : _params(p)
  , _image_size(s)
  , _T_kf(Matrix44::Identity())
{
  if(_params.numPyramidLevels <= 0)
  {
    _params.numPyramidLevels = 1 + std::round(
        std::log2(std::min(s.rows, s.cols) / (double) _params.minImageDimensionForPyramid));
  }

  _vo_pose = make_unique<VisualOdometryPoseEstimator>(K, b, _params);
  _kf_candidate = make_unique<KeyFrameCandidate>(p);
  _desc_pyr = make_unique<DenseDescriptorPyramid>(p);
}

VisualOdometryWithKeyFraming::~VisualOdometryWithKeyFraming() {}

Result VisualOdometryWithKeyFraming::
addFrame(const uint8_t* image_ptr, const float* disparity_ptr)
{
  const auto I = ToOpenCV(image_ptr, _image_size);
  const auto D = ToOpenCV(disparity_ptr, _image_size);

  _desc_pyr->init(I);

  if(!_vo_pose->hasTemplate())
  {
    Result ret;
    ret.pose.setIdentity();
    ret.covariance.setIdentity();
    ret.isKeyFrame = true;
    ret.keyFramingReason = KeyFramingReason::kFirstFrame;
    ret.optimizerStatistics.resize(_desc_pyr->size());
    for(auto& s : ret.optimizerStatistics) {
      s.numIterations = 0;
      s.finalError = 0.0f;
      s.firstOrderOptimality = 0.0f;
      s.status = PoseEstimationStatus::kFunctionTolReached;
    }

    _vo_pose->setTemplate(*_desc_pyr, D);
    return ret;
  }

  Matrix44 T_est;

  Result ret;
  ret.optimizerStatistics = _vo_pose->estimatePose(*_desc_pyr, _T_kf, T_est);
  ret.keyFramingReason = shouldKeyFrame(T_est);
  ret.isKeyFrame = ret.keyFramingReason != KeyFramingReason::kNoKeyFraming;

  if(!ret.isKeyFrame)
  {
    dprintf("updating kfc\n");
    _kf_candidate->set(*_desc_pyr, D);
    ret.pose = T_est * _T_kf.inverse();
    _T_kf = T_est;
  }
  else
  {

    dprintf("keyframe\n");

    if(_kf_candidate->empty())
    {
      _vo_pose->setTemplate(*_desc_pyr, D);
      ret.pose = T_est * _T_kf.inverse();
      _T_kf.setIdentity();
    }
    else
    {
      printf("\nrecompute\n");
      ret.optimizerStatistics = _vo_pose->setTemplate(
          _kf_candidate->_desc_pyr, _kf_candidate->_disparity).
          estimatePose(*_desc_pyr, Matrix44::Identity(), T_est);

      ret.pose = T_est;
      _T_kf = T_est;

      _kf_candidate->clear();
    }
  }

  _T_est = ret.pose;

  return ret;
}

int VisualOdometryWithKeyFraming::numPointsAtLevel(int level) const
{
  if(level < 0)
    level = _params.maxTestLevel;

  return _vo_pose->_tdata_pyr[level]->numPoints();
}

KeyFramingReason
VisualOdometryWithKeyFraming::shouldKeyFrame(const Matrix44& pose) const
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

  const auto& w = _vo_pose->_pose_estimator.getWeights();
  const auto thresh = _params.goodPointThreshold;
  auto num_good = std::count_if(std::begin(w), std::end(w),
                                [=](float w_i) { return w_i > thresh; });
  auto frac_good = num_good / (float) w.size();
  if(frac_good < _params.maxFractionOfGoodPointsToKeyFrame)
  {
    dprintf("kSmallFracOfGoodPoints\n");
    return KeyFramingReason::kSmallFracOfGoodPoints;
  }

  return KeyFramingReason::kNoKeyFraming;
}

} // bpvo

