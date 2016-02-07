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
}; // VisualOdometry

}; // bpvo

#endif // BPVO_VO_IMPL_H
