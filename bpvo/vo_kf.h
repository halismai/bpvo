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

#ifndef BPVO_VO_KF_H
#define BPVO_VO_KF_H

#include <bpvo/types.h>
#include <bpvo/pose_estimator_gn.h>
#include <bpvo/template_data.h>

namespace bpvo {

class VisualOdometryFrame;
class VisualOdometryPoseEstimator;
class DenseDescriptorPyramid;

class VisualOdometryWithKeyFraming
{
 public:
  /**
   * \param K the intrinsics matrix for the rectified image
   * \param b the stereo baseline
   * \param ImageSize size of the image at the finest resolution
   * \param AlgorithmParameters see types.h
   */
  VisualOdometryWithKeyFraming(const Matrix33& K, const float baseline,
                               ImageSize, AlgorithmParameters);

  ~VisualOdometryWithKeyFraming();

  /**
   */
  Result addFrame(const uint8_t* image_ptr, const float* disparity_ptr);

  /**
   * \return the number of points at the specified pyramid level
   */
  int numPointsAtLevel(int level = -1) const;

 private:
  AlgorithmParameters _params;
  ImageSize _image_size;
  UniquePointer<VisualOdometryPoseEstimator> _vo_pose;
  Matrix44 _T_kf;  //< motion wrt to the keyframe
  Matrix44 _T_est; //< last estimated motion

  struct KeyFrameCandidate;
  /**
   * Store information from the most recent successfuly tracked frame for
   * keyframing. When the keyframing criteria is met, data inside _kf_candidate
   * will be used as the kyeframe
   */
  UniquePointer<KeyFrameCandidate> _kf_candidate;

  /**
   * A pyramid of dense descriptors per level, decided from AlgorithmParameters
   */
  UniquePointer<DenseDescriptorPyramid> _desc_pyr;

 private:
  /**
   */
  KeyFramingReason shouldKeyFrame(const Matrix44& pose) const;
}; // VisualOdometryWithKeyFraming

}; // bpvo

#endif // BPVO_VO_KF_H
