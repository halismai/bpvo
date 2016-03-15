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

namespace bpvo {

struct VisualOdometryKf::KeyFrameCandidate
{
  UniquePointer<ImagePyramid> image_pyramid;
  cv::Mat disparity_map;

  inline bool empty() const { return disparity_map.empty(); }
  inline void clear() { disparity_map = cv::Mat(); }
}; // KeyFrameCandidate

VisualOdometryKf::VisualOdometryKf(const Matrix33& K, float b, ImageSize imsize,
                                   AlgorithmParameters params)
    : _params(params)
    , _vo_pose(make_unique<VisualOdometryPose>(K, b, imsize, params))
    , _kf_point_cloud(make_unique<PointCloud>())
    , _kf_candidate(make_unique<KeyFrameCandidate>()) {}

VisualOdometryKf::~VisualOdometryKf() {}

Result VisualOdometryKf::addFrame(const uint8_t* image_ptr, const float* dmap_ptr)
{
  Result ret;

  if(_vo_pose->hasData()) {
    dprintf("estimating pose\n");
    ret = _vo_pose->estimatePose(image_ptr, _T_kf);
    dprintf("vo_pose got %zu\n", ret.optimizerStatistics.size());
  } else {
    dprintf("setting first frame\n");
    //
    // the first frame, there is no pose estimation
    //
    ret.optimizerStatistics.resize( _vo_pose->getImagePyramid().size() );
    for(auto& s : ret.optimizerStatistics) {
      s.numIterations = 0;
      s.finalError = 0.0;
      s.firstOrderOptimality = 0.0;
      s.status = PoseEstimationStatus::kFunctionTolReached;
    }

    ret.keyFramingReason = KeyFramingReason::kFirstFrame;
  }

  dprintf("template\n");
  _vo_pose->setTemplate(image_ptr, dmap_ptr);

  return ret;
}

} // bpvo

