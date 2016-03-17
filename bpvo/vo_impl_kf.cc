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

#include "bpvo/vo_impl_kf.h"
#include "bpvo/vo_pose.h"
#include "bpvo/point_cloud.h"
#include "bpvo/image_pyramid.h"

#include <opencv2/highgui/highgui.hpp>

namespace bpvo {

void ShowImages(const cv::Mat& I0, const cv::Mat& I1)
{
  cv::Mat D;
  cv::absdiff(I0, I1, D);
  cv::imshow("image", D);
  cv::waitKey(0);
}

struct VisualOdometryKf::KeyFrameCandidate
{
  UniquePointer<ImagePyramid> image_pyramid;
  cv::Mat disparity_map;

  inline bool empty() const { return disparity_map.empty(); }
  inline void clear() { disparity_map = cv::Mat(); }

  inline void setData(const ImagePyramid& pyr, const float* d_ptr,
                      const ImageSize& image_size,
                      bool clone = true)
  {
    image_pyramid.reset(new ImagePyramid(pyr));

    const cv::Mat D(image_size.rows, image_size.cols, cv::DataType<float>::type, (void*) d_ptr);
    disparity_map = clone ? D.clone() : D;
  }

  KeyFrameCandidate() {}
}; // KeyFrameCandidate

VisualOdometryKf::VisualOdometryKf(const Matrix33& K, float b, ImageSize imsize,
                                   AlgorithmParameters params)
    : _params(params)
    , _vo_pose(make_unique<VisualOdometryPose>(K, b, imsize, params))
    , _kf_point_cloud(make_unique<PointCloud>())
    , _kf_candidate(make_unique<KeyFrameCandidate>()) {}

VisualOdometryKf::~VisualOdometryKf() {}

static void setFirstFrameResult(Result& ret, int n_levels)
{
  ret.optimizerStatistics.resize(n_levels);
  for(auto& s : ret.optimizerStatistics) {
    s.numIterations = 0;
    s.finalError = 0.0;
    s.firstOrderOptimality = 0.0;
    s.status = PoseEstimationStatus::kFunctionTolReached;
  }
  ret.keyFramingReason = KeyFramingReason::kFirstFrame;
}

Result VisualOdometryKf::addFrame(const uint8_t* image_ptr, const float* dmap_ptr)
{
  Result ret;

  if(!_vo_pose->hasData())
  {
    ret.pose.setIdentity();
    setFirstFrameResult(ret, _vo_pose->getImagePyramid().size());
    _vo_pose->setTemplate(image_ptr, dmap_ptr);
    return ret;
  }

  Matrix44 T_est;
  ret = _vo_pose->estimatePose(image_ptr, _T_kf, T_est);
  ret.keyFramingReason = shouldKeyFrame(ret);
  ret.isKeyFrame = ret.keyFramingReason != KeyFramingReason::kNoKeyFraming;

  if(!ret.isKeyFrame)
  {
    if(true) //TODO test if the disparity is good for KeyFrameCandidate
    {
      _kf_candidate->setData(_vo_pose->getImagePyramid(), dmap_ptr, _vo_pose->imageSize());
    }
    else
    {
      _kf_candidate->clear();
    }

    ret.pose = T_est * _T_kf.inverse();
    _T_kf = T_est;

    std::cout << "POSE:\n" << ret.pose << std::endl;
  }
  else
  {
    if(_kf_candidate->empty())
    {
      _vo_pose->setTemplate(_vo_pose->getImagePyramid(), dmap_ptr);
      Warn("no KEYFRAME CANDIDATE\n");
      std::cout << _T_kf << std::endl;
      std::cout << T_est << std::endl;
      T_est = T_est * _T_kf.inverse();
    }
    else
    {
      _vo_pose->setTemplate(*_kf_candidate->image_pyramid,
                            _kf_candidate->disparity_map.ptr<const float>());
      auto tmp = _vo_pose->estimatePose(image_ptr, Matrix44::Identity(), T_est);
      _kf_candidate->clear();

      std::cout << tmp << std::endl;

      Warn("kf pose\n");
      std::cout << T_est << std::endl;
      Warn("----\n");

      const cv::Mat I(480, 640, cv::DataType<uint8_t>::type, (void*) image_ptr);
      ShowImages(_kf_candidate->image_pyramid->operator[](0), I);
    }

    ret.pose = T_est;
    _T_kf.setIdentity();
  }

  return ret;
}

KeyFramingReason VisualOdometryKf::shouldKeyFrame(const Result& result)
{
  if(result.keyFramingReason == KeyFramingReason::kFirstFrame)
  {
    return KeyFramingReason::kFirstFrame;
  }

  auto t_norm = result.pose.block<3,1>(0,3).squaredNorm();
  if(t_norm > math::sq(_params.minTranslationMagToKeyFrame))
  {
    return KeyFramingReason::kLargeTranslation;
  }

  auto r_norm = math::RotationMatrixToEulerAngles(result.pose).squaredNorm();
  if(r_norm > math::sq(_params.minRotationMagToKeyFrame))
  {
    return KeyFramingReason::kLargeRotation;
  }

  const auto& w = _vo_pose->getWeights();
  const auto thresh = _params.goodPointThreshold;
  auto num_good = std::count_if(std::begin(w), std::end(w),
                                [=](float w_i) { return w_i > thresh; });
  auto frac_good = num_good / (float) w.size();
  if(frac_good < _params.maxFractionOfGoodPointsToKeyFrame)
  {
    return KeyFramingReason::kSmallFracOfGoodPoints;
  }

  return KeyFramingReason::kNoKeyFraming;
}

} // bpvo

