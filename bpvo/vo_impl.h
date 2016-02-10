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

#ifndef BPVO_VO_IMPL_H
#define BPVO_VO_IMPL_H

#include <bpvo/types.h>
#include <bpvo/warps.h>
#include <bpvo/pose_estimator_gn.h>
#include <bpvo/mestimator.h>
#include <bpvo/vo.h>
#include <bpvo/template_data_.h>
#include <bpvo/channels.h>
#include <bpvo/linear_system_builder.h>

#include <vector>

namespace cv {
class Mat;
}; // cv

namespace bpvo {

class VisualOdometry::Impl
{
  friend class VisualOdometry;

 public:
  typedef RawIntensity ChannelsT;
  typedef RigidBodyWarp WarpT;
  typedef TemplateData_<ChannelsT, WarpT> TData;
  typedef PoseEstimatorGN<TData> PoseEstimatorT_;
  typedef PoseEstimatorBase<PoseEstimatorT_> PoseEstimatorT;

  typedef UniquePointer<TData> TDataPointer;
  typedef std::vector<TDataPointer> TDataPyramid;

  typedef VisualOdometry::PointVector PointVector;

  /**
   * \param K the intrinsic calibration matrix
   * \param baseline stereo baseline
   * \param ImageSize image size at the finest pyramid level
   * \param AlgorithmParameters parameters
   */
  Impl(const Matrix33& K, const float& baseline, ImageSize, AlgorithmParameters);

  virtual ~Impl();

  /**
   */
  Result addFrame(const uint8_t* image_ptr, const float* disparity_map_ptr);

  /**
   */
  int numPointsAtLevel(int) const;

  /**
   */
  const PointVector& pointsAtLevel(int) const;

 protected:
  AlgorithmParameters _params;
  PoseEstimatorParameters _pose_est_params;
  PoseEstimatorParameters _pose_est_params_low_res;
  ImageSize _image_size;

  PoseEstimatorT _pose_estimator;
  std::vector<TDataPointer> _tdata_pyr;
  std::vector<ChannelsT> _channels_pyr;

  Matrix44 _T_kf;

 protected:
  void setAsKeyFrame(const std::vector<ChannelsT>&, const cv::Mat&);

  Matrix44 estimatePose(const std::vector<ChannelsT>&, const Matrix44& T_init,
                        std::vector<OptimizerStatistics>& stats);

  /**
   * keyframing based on pose and valid points
   */
  KeyFramingReason shouldKeyFrame(const Matrix44&, const std::vector<float>& weights);


  struct KeyFrameCandidate
  {
    std::vector<ChannelsT> channels_pyr;
    cv::Mat disparity;

    /**
     * \return true if the KeyFrameCandidate is is empty
     */
    bool empty() const;

    /**
     * clears the data (calling empty() after this will return true)
     */
    void clear();
  }; // KeyFrameCandidate

  void setAsKeyFrame(const KeyFrameCandidate& kfc) {
    this->setAsKeyFrame(kfc.channels_pyr, kfc.disparity);
  }

  KeyFrameCandidate _kf_candidate;

}; // VisualOdometry

}; // bpvo

#endif // BPVO_VO_IMPL_H
