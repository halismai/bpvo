#include "bpvo/vo_impl.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

namespace bpvo {

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

VisualOdometry::Impl::Impl(const Matrix33& K, const float& b, ImageSize image_size,
                           AlgorithmParameters params)
    : _params(params)
    , _pose_est_params(params)
    , _pose_est_params_low_res(params)
    , _image_size(image_size)
    , _T_kf(Matrix44::Identity())
{
  assert( _image_size.numel() > 0 );

  //
  // set auto number of pyramid levels if needed
  //
  if(_params.numPyramidLevels <= 0)
    _params.numPyramidLevels = getNumberOfPyramidLevels(_image_size, 40);

  if(_params.relaxTolerancesForCoarseLevels)
    _pose_est_params_low_res.relaxTolerance();


  Matrix33 K_pyr(K);
  float b_pyr = b;
  _tdata_pyr.push_back(make_unique<TData>(K_pyr, b_pyr, 0));
  for(int i = 1; i < _params.numPyramidLevels; ++i) {
    K_pyr *= 0.5; K_pyr(2,2) = 1.0f;
    b_pyr *= 2.0;
    _tdata_pyr.push_back(make_unique<TData>(K_pyr, b_pyr, i));
  }

  for(int i= 0; i < _params.numPyramidLevels; ++i)
    _channels_pyr.push_back(ChannelsT(_params.sigmaPriorToCensusTransform,
                                      _params.sigmaBitPlanes));
}

VisualOdometry::Impl::~Impl() {}

Result VisualOdometry::Impl::addFrame(const uint8_t* I_ptr, const float* D_ptr)
{
  assert( !_channels_pyr.empty() && !_tdata_pyr.empty() );
  assert( _channels_pyr.size() == _tdata_pyr.size() );

  cv::Mat I;
  ToOpenCV(I_ptr, _image_size).copyTo(I);

  _channels_pyr.front().compute(I);
  for(size_t i = 1; i < _channels_pyr.size(); ++i) {
    cv::pyrDown(I, I);
    _channels_pyr[i].compute(I);
  }

  Result ret;
  ret.optimizerStatistics.resize(_channels_pyr.size());

  if(0 == _tdata_pyr.front()->numPoints()) {
    //
    // the first frame added to vo
    //
    cv::Mat D = ToOpenCV(D_ptr, _image_size);
    this->setAsKeyFrame(_channels_pyr, D);
    ret.isKeyFrame = true;
    ret.pose = _T_kf;
    ret.covariance.setIdentity();
    ret.keyFramingReason = KeyFramingReason::kFirstFrame;
    return ret;
  }

  //
  // initialize pose estimation for the next frame using the latest estimate
  //
  Matrix44 T_est = _T_kf;
  _pose_estimator.setParameters(_pose_est_params_low_res);
  for(int i = _channels_pyr.size()-1; i >= 1; --i) {
    ret.optimizerStatistics[i] = _pose_estimator.run(_tdata_pyr[i].get(), _channels_pyr[i], T_est);
  }

  // process the finest pyramid level
  _pose_estimator.setParameters(_pose_est_params);
  ret.optimizerStatistics.front() =
      _pose_estimator.run(_tdata_pyr.front().get(), _channels_pyr.front(), T_est);

  //
  // TODO test for keyframing
  //
  ret.isKeyFrame = false;
  ret.pose = T_est * _T_kf.inverse(); // return the relative motion wrt to the added frame
  _T_kf = T_est; // replace the initialization with the new motion

  return ret;
}

void VisualOdometry::Impl::setAsKeyFrame(const std::vector<ChannelsT>& cn,
                                         const cv::Mat& disparity)
{
  assert( _tdata_pyr.size() == cn.size() );

  for(size_t i = 0; i < cn.size(); ++i)
    _tdata_pyr[i]->setData(cn[i], disparity);
}

int VisualOdometry::Impl::numPointsAtLevel(int level) const
{
  assert( level >= 0 && level < (int) _tdata_pyr.size() );

  return _tdata_pyr[level]->numPoints();
}

auto VisualOdometry::Impl::pointsAtLevel(int level) const -> const PointVector&
{
  assert( level >= 0 && level < (int) _tdata_pyr.size() );

  return _tdata_pyr[level]->points();
}

} // bpvo
