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
#include <bpvo/trajectory.h>
#include <bpvo/point_cloud.h>
#include <bpvo/rigid_body_warp.h>
#include <bpvo/disparity_space_warp.h>
#include <bpvo/image_pyramid.h>

#include <vector>

namespace cv {
class Mat;
}; // cv

namespace bpvo {

class VisualOdometry::Impl
{
  friend class VisualOdometry;

 public:

#if defined(WITH_BITPLANES)
  typedef BitPlanes ChannelsT;
#else
  typedef RawIntensity ChannelsT;
#endif

#if defined(WITH_DISPARITY_SPACE_WARP)
  typedef DisparitySpaceWarp WarpT;
#else
  typedef RigidBodyWarp WarpT;
#endif

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

  /**
   */
  const WeightsVector& getWeights() const;

  /**
   */
  inline const Trajectory& trajectory() const { return _trajectory; }

 protected:
  /** AlgorithmParameters */
  AlgorithmParameters _params;

  /** PoseEstimatorParameters for fine pyramid levels (high res) */
  PoseEstimatorParameters _pose_est_params;

  /** PoseEstimatorParameters for coarse pyramid levels (low res) */
  PoseEstimatorParameters _pose_est_params_low_res;

  /** the image size */
  ImageSize _image_size;

  /** pose estimator */
  PoseEstimatorT _pose_estimator;

  /** TemplateData _pyr pyramid level */
  std::vector<TDataPointer> _tdata_pyr;

  /** image pyramid */
  UniquePointer<ImagePyramid> _image_pyramid;

  /** buffer to contain input data pyr pyramid level */
  std::vector<ChannelsT> _channels_pyr;

  /** Transformation wrt to the keyframe, used to initialize the pose for new frames */
  Matrix44 _T_kf;

  /** trajectory since the first call to addFrame */
  Trajectory _trajectory;

  /**
   * the point cloud corresponding to the current keyframe
   */
  PointCloud _kf_point_cloud;

  /**
   * used to get the pose of the point cloud from the trajectory
   *
   * T_pc_kf = _trajectory[ _kf_pose_index ];
   */
  int _kf_pose_index = 0;
  int _frame_index = 0;

  /**
   * we keep this here for point cloud colorization
   */
  cv::Mat _input_image;

  /**
   * point and their weights from the last keyframe
   */
  PointVector _pts_with_weights;

 protected:
  void setAsKeyFrame(const std::vector<ChannelsT>&, const cv::Mat&);

  /**
   * Estimate the pose of the input channels with respect to the current keyframe
   *
   * \param input_channels the input channels
   * \param T_init         pose initialization
   * \param stats          optimization stats per pyramid level
   *
   * \return the estimated pose
   */
  Matrix44 estimatePose(const std::vector<ChannelsT>& input_channels,
                        const Matrix44& T_init,
                        std::vector<OptimizerStatistics>& stats);

  /**
   * keyframing based on pose and valid points
   */
  KeyFramingReason shouldKeyFrame(const Matrix44&, const WeightsVector& weights);


  /**
   * Holds the data need to set a new keyframe
   */
  struct KeyFrameCandidate
  {
    std::vector<ChannelsT> channels_pyr; // the channels
    cv::Mat disparity;            // the disparity image

    /**
     * \return true if the KeyFrameCandidate is is empty
     */
    bool empty() const;

    /**
     * clears the data (calling empty() after this will return true)
     */
    void clear();
  }; // KeyFrameCandidate

  KeyFrameCandidate _kf_candidate;

  inline void setAsKeyFrame(const KeyFrameCandidate& kfc) {
    this->setAsKeyFrame(kfc.channels_pyr, kfc.disparity);
  }

  /**
   * a wrapper to create template data given calibration and pyramid level
   */
  UniquePointer<TData> makeTemplateData(const Matrix33& K, float b, int pyr_level) const;

  inline bool isFirstFrame() const
  {
    return _tdata_pyr[_params.maxTestLevel]->numPoints() == 0;
  }
}; // VisualOdometry

}; // bpvo

#endif // BPVO_VO_IMPL_H
