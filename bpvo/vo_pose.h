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

#ifndef BPVO_VO_POSE_H
#define BPVO_VO_POSE_H

#include <bpvo/types.h>
#include <bpvo/pose_estimator_gn.h>
#include <bpvo/template_data.h>
#include <bpvo/image_pyramid.h>

namespace cv {
class Mat;
}; // cv

namespace bpvo {

/**
 * Performs pose estimation for VO. Does not handle keyframing
 */
class VisualOdometryPose
{
  typedef PoseEstimatorBase<PoseEstimatorGN<TemplateData>> PoseEstimatorT;

 public:
  /**
   */
  VisualOdometryPose(const Matrix33& K, const float baseline, ImageSize, AlgorithmParameters);

  /**
   * sets the template data (keyframe)
   */
  void setTemplate(const uint8_t* image_ptr, const float* disparity_map_ptr);

  void setTemplate(const ImagePyramid&, const float*);

  /**
   * Estimates the pose wrt the input image
   */
  Result estimatePose(const uint8_t* image_ptr, const Matrix44& T_init, Matrix44& T_est);

  /**
   * \return true if there is template data
   */
  bool hasData() const;

  /**
   * \return the image pyramid (check _image_pyramid.front().empty()) to verify
   * that the there are images
   */
  inline const ImagePyramid& getImagePyramid() const { return _image_pyramid; }

  /**
   * \return the M-Estimator weights computed at the most recent call to
   * estimatePose
   */
  inline const WeightsVector& getWeights() const { return _pose_estimator.getWeights(); }

  /**
   * \return the image size
   */
  inline const ImageSize& imageSize() const { return _image_size; }

 protected:
  ImageSize _image_size;
  AlgorithmParameters _params;
  PoseEstimatorParameters _pose_est_params, _pose_est_params_low_res;
  PoseEstimatorT _pose_estimator;
  ImagePyramid _image_pyramid;
  std::vector<UniquePointer<TemplateData>> _tdata_pyr;
}; // VisualOdometryPose

}; // bpvo

#endif // BPVO_VO_POSE_H
