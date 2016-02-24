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

#ifndef BPVO_TEMPLATE_DATA__H
#define BPVO_TEMPLATE_DATA__H

#include <bpvo/types.h>
#include <bpvo/debug.h>
#include <bpvo/utils.h>
#include <bpvo/imgproc.h>
#include <bpvo/math_utils.h>
#include <bpvo/interp_util.h>
#include <bpvo/parallel.h>

#include <opencv2/core/core.hpp>

#include <utility>
#include <iostream>
#include <fstream>

#if defined(WITH_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

namespace bpvo {

template <class ChannelsType, class WarpType>
class TemplateData_
{
 public:
  typedef WarpType                          Warp;
  typedef typename Warp::Point              Point;
  typedef typename Warp::Jacobian           Jacobian;
  typedef typename Warp::WarpJacobian       WarpJacobian;
  typedef typename Warp::PointVector        PointVector;
  typedef typename Warp::JacobianVector     JacobianVector;
  typedef typename Warp::WarpJacobianVector WarpJacobianVector;

  typedef ChannelsType                   Channels;
  typedef typename Channels::PixelVector PixelVector;

  static constexpr int NumChannels = Channels::NumChannels;

 public:
  /**
   * \param K the intrinsics matrix
   * \param baseline the stereo baseline
   * \param pyr_level the pyramid level
   * \param min_pixels_for_nms minimum number of pixels to do NMS
   * \param min_saliency minimum saliency for a pixel to be used
   * \param min_disparity minimum disparity for a pixel to be used
   * \param max_disparity maximum disparity for a pixel to be used
   * \param non maxima suppresion radius
   */
  inline TemplateData_(const Matrix33& K,
                       float baseline,
                       int pyr_level,
                       int min_pixels_for_nms = 320*240,
                       float min_saliency = 0.01f,
                       float min_disparity = 1.0f,
                       float max_disparity = 512.0f,
                       int nms_radius = 1,
                       bool with_normalization = true)
      : _warp(K, baseline)
        , _pyr_level(pyr_level)
        , _min_pixels_for_nms(min_pixels_for_nms)
        , _min_saliency(min_saliency)
        , _min_disparity(min_disparity)
        , _max_disparity(max_disparity)
        , _nms_radius(nms_radius)
        , _with_normalization(with_normalization)
  {
    assert( _pyr_level >= 0 );
    assert( nms_radius > 0 );
  }

  inline ~TemplateData_() {}

 public:
  /**
   * set the data, i.e. compute points, pixels and jacobians
   */
  void setData(const Channels& channels, const cv::Mat& disparity);

  /**
   * computes the residuals given the input channels and pose
   */
  void computeResiduals(const Channels& channels, const Matrix44& pose,
                        ResidualsVector& residuals, ValidVector& valid);

  /**
   * reserves memory for 'n' points
   */
  inline void reserve(size_t n)
  {
    _jacobians.reserve(n * NumChannels);
    _points.reserve(n);
    _pixels.reserve(n * NumChannels);
  }

  /** \return the number of points */
  inline int numPoints() const { return _points.size(); }

  /** \return the number of pixels */
  inline int numPixels() const { return _pixels.size(); }

  /**
   * \return the i-th point
   */
  const typename PointVector::value_type& X(size_t i) const { return _points[i]; }
  typename PointVector::value_type& X(size_t i) { return _points[i]; }

  /**
   * \return the i-th jacobian
   */
  const typename JacobianVector::value_type& J(size_t i) const { return _jacobians[i]; }
  typename JacobianVector::value_type& J(size_t i) { return _jacobians[i]; }

  /**
   * \return the i-th pixel value
   */
  const typename PixelVector::value_type& I(size_t i) const { return _pixels[i]; }
  typename PixelVector::value_type& I(size_t i) { return _pixels[i]; }

  inline const WarpBase<Warp>& warp() const { return _warp; }

  inline const JacobianVector& jacobians() const { return _jacobians; }
  inline const PointVector& points() const { return _points; }

  inline const PixelVector& pixels() const { return _pixels; }
  inline       PixelVector& pixels()       { return _pixels; }

 protected:
  Warp _warp;

  int _pyr_level;
  int _min_pixels_for_nms;
  float _min_saliency;
  float _min_disparity;
  float _max_disparity;
  int _nms_radius;
  bool _with_normalization;

  JacobianVector _jacobians;
  PointVector _points;
  PixelVector _pixels;

  inline void clear()
  {
    _jacobians.clear();
    _points.clear();
    _pixels.clear();
  }

  inline void resize(size_t n)
  {
    _jacobians.resize( n * NumChannels );
    _points.resize( n );
    _pixels.resize( n * NumChannels );
  }

  cv::Mat_<float> _buffer; // buffer to compute saliency maps
  std::vector<int> _inds;  // linear indices into valid points

  /**
   */
  void getValidPoints(const Channels& cn, const cv::Mat& D);

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}; // TemplateData_


template <class CN, class W> inline
void TemplateData_<CN,W>::getValidPoints(const CN& cn, const cv::Mat& D)
{
  //
  // clear the  revious points
  //
  _points.clear();
  _inds.clear();

  const ImageSize image_size(cn.rows(), cn.cols());
  IsLocalMax<float> is_local_max(nullptr, cn.cols(), -1);

  //
  // we'll pre-compute the saliency maps anyways because this is faster with
  // BitPlanes, but if we are doing intensity only we are better off computing
  // the saliency on the fly if nms is not neededq
  //
  cn.computeSaliencyMap(_buffer);

  bool do_nms = image_size.rows * image_size.cols >= _min_pixels_for_nms;
  if(do_nms)
  {
    //
    // set the data for NMS
    //
    is_local_max.setPointer(_buffer.ptr<float>());
    is_local_max.setStride(_buffer.cols);
    is_local_max.setRadius(_nms_radius);
  }

  const int border = std::max(2, _nms_radius);

  //
  // we'll split this in two loops for cache & branch prediction friendlyness
  //
  std::vector<uint16_t> tmp_inds;
  tmp_inds.reserve( image_size.rows * image_size.cols * 0.25 );

  int max_rows = image_size.rows - border - 1,
      max_cols = image_size.cols - border - 1;

  // no benefit more openmp here
  for(int y = border; y < max_rows; ++y)
  {
    auto s_row = _buffer.ptr<float>(y);
    for(int x = border; x < max_cols; ++x)
    {
      if(s_row[x] <= _min_saliency || !is_local_max(y,x))
        continue;

      {
        tmp_inds.push_back(y);
        tmp_inds.push_back(x);
      }
    }
  }

  _inds.reserve(tmp_inds.size()/2);
  _points.reserve(tmp_inds.size()/2);
  auto D_ptr = D.ptr<float>();
  for(size_t i = 0; i < tmp_inds.size(); i += 2)
  {
    int y = static_cast<int>( tmp_inds[i + 0] );
    int x = static_cast<int>( tmp_inds[i + 1] );
    int ii = y*image_size.cols + x;
    float d = D_ptr[ (1<<_pyr_level)*(y*D.cols + x) ];
    if(d > _min_disparity && d < _max_disparity)
    {
      _points.push_back( _warp.makePoint(x, y, d) );
      _inds.push_back( ii );
    }
  }

  if(_with_normalization)
    _warp.setNormalization(_points);
}

namespace {

template <class TemplateDataT>
struct SetTemplateDataBody : public ParallelForBody
{
  typedef typename TemplateDataT::Channels Channels;
  typedef typename TemplateDataT::Warp     WarpType;
  typedef typename TemplateDataT::JacobianVector JacobianVector;
  typedef typename TemplateDataT::PointVector    PointVector;
  typedef typename TemplateDataT::PixelVector    PixelVector;

 public:
  SetTemplateDataBody(const Channels& channels, const PointVector& points,
                      const std::vector<int>& inds, const WarpType& warp,
                      PixelVector& pixels, JacobianVector& jacobians)
      : _channels(channels)
      , _points(points)
      , _inds(inds)
      , _warp(warp)
      , _pixels(pixels)
      , _jacobians(jacobians) {}

  inline void operator()(const Range& range) const
  {
    const int n = _points.size();
    const int stride = _channels.cols();

    for(int c = range.begin(); c != range.end(); ++c)
    {
      auto c_ptr = _channels.channelData(c);
      auto P_ptr = _pixels.data() + c*n;
      auto J_ptr = _jacobians.data() + c*n;

      for(int i = 0; i < n; ++i)
      {
        auto ii = _inds[i];
        P_ptr[i] = c_ptr[ii];
        float Ix = 0.5f * (c_ptr[ii+1] - c_ptr[ii-1]),
              Iy = 0.5f * (c_ptr[ii+stride] - c_ptr[ii-stride]);
        _warp.jacobian(_points[i], Ix, Iy, J_ptr[i].data());
      }
    }
  }

 protected:
  const Channels& _channels;
  const PointVector& _points;
  const std::vector<int>& _inds;
  const WarpType& _warp;
  PixelVector& _pixels;
  JacobianVector& _jacobians;
}; // SetTemplateDataBody

}; // namespace

template <class CN, class W> inline
void TemplateData_<CN,W>::setData(const Channels& channels, const cv::Mat& D)
{
  assert( D.type() == cv::DataType<float>::type );

  getValidPoints(channels, D);

  const auto N = _points.size();

  _pixels.resize(N * NumChannels);
  _jacobians.resize(N * NumChannels);

  SetTemplateDataBody<TemplateData_<CN,W>> func(channels, _points, _inds,
                                                *warp().derived(), _pixels, _jacobians);

  parallel_for(Range(0, NumChannels), func);

  // NOTE: we push an empty Jacobian at the end because of SSE code loading
  _jacobians.push_back(Jacobian::Zero());
}

template <class Channels>
class ComputeResidualsBody : public ParallelForBody
{
 public:
  ComputeResidualsBody(const Channels& cn, const BilinearInterp<float>& interp,
                       int num_points, const float* pixels, float* residuals)
      : _channels(cn)
      , _interp(interp)
      , _num_points(num_points)
      , _pixels(pixels)
      , _residuals(residuals) {}

  inline void operator()(const Range& range) const
  {
    for(int c = range.begin(); c != range.end(); ++c)
    {
      int off = c * _num_points;
      auto I0_ptr = _pixels + off;
      auto I1_ptr = _channels.channelData(c);
      auto r_ptr = _residuals + off;
      _interp.run(I0_ptr, I1_ptr, r_ptr);
    }
  }

 protected:
  const Channels& _channels;
  const BilinearInterp<float>& _interp;
  int _num_points;
  const float* _pixels;
  float* _residuals;
}; // computeResiduals

template <class CN, class W> inline
void TemplateData_<CN,W>::
computeResiduals(const Channels& channels, const Matrix44& pose,
                 ResidualsVector& residuals, ValidVector& valid)
{
  _warp.setPose(pose);

  BilinearInterp<float> interp;
  interp.init(_warp, _points, channels[0].rows, channels[0].cols);

  residuals.resize(_pixels.size());
  valid.resize(_pixels.size());

  ComputeResidualsBody<CN> func(channels, interp, numPoints(),
                                _pixels.data(), residuals.data());
  parallel_for(Range(0, NumChannels), func);

  valid.swap(interp.valid());
}

}; // bpvo

#endif // BPVO_TEMPLATE_DATA__H
