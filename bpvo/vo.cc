#include "bpvo/vo.h"
#include "bpvo/debug.h"
#include "bpvo/linear_system_builder.h"
#include "bpvo/linear_system_builder_2.h"
#include "bpvo/math_utils.h"
#include "bpvo/warps.h"
#include "bpvo/channels.h"
#include "bpvo/pose_estimator.h"
#include "bpvo/template_data_.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <memory>
#include <cmath>
#include <limits>
#include <vector>
#include <iostream>

#include <Eigen/Cholesky>

namespace bpvo {

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
 */
struct VisualOdometry::Impl
{
  typedef RawIntensity ChannelsT;
  typedef RigidBodyWarp WarpT;
  typedef TemplateData_<ChannelsT, WarpT> TData;
  typedef PoseEstimator<TData, LinearSystemBuilder> PoseEstimatorT;

  typedef UniquePointer<TData> TDataPointer;
  typedef std::vector<TDataPointer> TDataPyramid;

  AlgorithmParameters _params;         //< algorithm parameters
  PoseEstimatorParameters _pose_est_params;
  PoseEstimatorParameters _pose_est_params_low_res;

  ImageSize _image_size;               //< image size at the finest resolution
  TDataPyramid _tdata_pyr;             //< pyramid of template data
  PoseEstimatorT _pose_estimator;      //< the pose estimator
  std::vector<ChannelsT> _channels_pyr; //< the channels

  Matrix44 _T_kf; //< current estimate of pose from the last keyframe

  Impl(const Matrix33& K_, const float& b_, ImageSize image_size, AlgorithmParameters params)
      : _params(params), _image_size(image_size), _pose_estimator(params), _T_kf(Matrix44::Identity())
  {
    assert( _image_size.rows > 0 && _image_size.cols > 0 );

    // auto set the number of pyramid levels if requested
    if(_params.numPyramidLevels <= 0)
      _params.numPyramidLevels = getNumberOfPyramidLevels(_image_size, 40);

    _pose_est_params = PoseEstimatorParameters(_params);
    _pose_est_params_low_res = _pose_est_params;
    if(_params.relaxTolerancesForCoarseLevels)
      _pose_est_params_low_res.relaxTolerance();

    // create the pyramid. Note, _params_low_res is irrelevant here. Basically,
    // we should have a different struct for the optimization parameters
    Matrix33 K(K_);
    float b = b_;
    for(int i = 0; i < _params.numPyramidLevels; ++i) {
      _tdata_pyr.push_back(make_unique<TData>(K, b, i));
      b *= 2.0;
      K *= 0.5f; K(2,2) = 1.0f;
    }

    for(int i = 0; i < _params.numPyramidLevels; ++i) {
      _channels_pyr.push_back(ChannelsT(_params.sigmaPriorToCensusTransform,
                                        _params.sigmaBitPlanes));
    }
  }

  inline ~Impl() {}

  Result addFrame(const uint8_t*, const float*);
  void setAsKeyFrame(const std::vector<ChannelsT>&, const cv::Mat&);
}; // Impl

VisualOdometry::VisualOdometry(const Matrix33& K, float baseline,
                               ImageSize image_size, AlgorithmParameters params)
    : _impl(new Impl(K, baseline, image_size, params)) {}

VisualOdometry::~VisualOdometry() { delete _impl; }

Result VisualOdometry::addFrame(const uint8_t* image, const float* disparity)
{
  return _impl->addFrame(image, disparity);
}


int VisualOdometry::numPointsAtLevel(int level) const
{
  return _impl->_tdata_pyr[level]->numPoints();
}

auto VisualOdometry::pointsAtLevel(int level) const -> const PointVector&
{
  return _impl->_tdata_pyr[level]->points();
}

Result VisualOdometry::Impl::addFrame(const uint8_t* I_ptr, const float* D_ptr)
{
  cv::Mat I, D;
  ToOpenCV(I_ptr, _image_size).copyTo(I);
  ToOpenCV(D_ptr, _image_size).copyTo(D);

  //
  // compute the channels
  //
  _channels_pyr[0].compute(I);
  for(size_t i = 1; i < _channels_pyr.size(); ++i) {
    cv::pyrDown(I, I);
    _channels_pyr[i].compute(I);
  }

  Result ret;
  ret.optimizerStatistics.resize(_channels_pyr.size());
  if(0 == _tdata_pyr.front()->numPoints()) {
    // the first frame
    setAsKeyFrame(_channels_pyr, D);
    ret.isKeyFrame = true;
    ret.pose = _T_kf;
    ret.covariance.setIdentity();
    ret.keyFramingReason = KeyFramingReason::kFirstFrame;
    return ret;
  }

  Matrix44 T_est = _T_kf;
  _pose_estimator.setParameters(_pose_est_params_low_res);
  for(int i = _channels_pyr.size()-1; i >= 0; --i) {
    if(i == 0)
      _pose_estimator.setParameters(_pose_est_params);
    auto stats = _pose_estimator.run(_tdata_pyr[i].get(), _channels_pyr[i], T_est);
    ret.optimizerStatistics[i] = stats;
  }

  ret.isKeyFrame = false;
  ret.pose = T_est * _T_kf.inverse();
  _T_kf = T_est;

  return ret;
}

void VisualOdometry::Impl::setAsKeyFrame(const std::vector<ChannelsT>& cn,
                                         const cv::Mat& disparity)
{
  assert( _tdata_pyr.size() == cn.size() );

  for(size_t i = 0; i < cn.size(); ++i) {
    _tdata_pyr[i]->setData(cn[i], disparity);
  }
}

}; // bpvo

