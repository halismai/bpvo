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

#ifndef BPVO_VO_IMPL_KF_H
#define BPVO_VO_IMPL_KF_H

#include <bpvo/types.h>

namespace bpvo {

class VisualOdometryPose;
class PointCloud;

class VisualOdometryKf
{
 public:
  VisualOdometryKf(const Matrix33&, float b, ImageSize, AlgorithmParameters);
  ~VisualOdometryKf();

  Result addFrame(const uint8_t*, const float*);

  int numPointsAtLevel(int level = -1) const;

 protected:
  AlgorithmParameters _params;
  ImageSize _image_size;
  UniquePointer<VisualOdometryPose> _vo_pose;
  UniquePointer<PointCloud> _kf_point_cloud;

  Matrix44 _T_kf = Matrix44::Identity();

  struct KeyFrameCandidate;
  UniquePointer<KeyFrameCandidate> _kf_candidate;

 private:
  KeyFramingReason shouldKeyFrame(const Result&);
}; // VisualOdometryKf

}; // bpvo

#endif // BPVO_VO_IMPL_KF_H
