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
#include <bpvo/pixel_selector.h>

#include <opencv2/core/core.hpp>

#include <utility>
#include <iostream>
#include <fstream>

#if defined(WITH_TBB)
#define TEMPLATE_DATA_SET_DATA_WITH_TBB 0
#if TEMPLATE_DATA_SET_DATA_WITH_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif
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
                       int nms_radius = 1)
      : _warp(K, baseline)
        , _pyr_level(pyr_level)
        , _min_pixels_for_nms(min_pixels_for_nms)
        , _min_saliency(min_saliency)
        , _min_disparity(min_disparity)
        , _max_disparity(max_disparity)
        , _nms_radius(nms_radius)
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
                        std::vector<float>& residuals, std::vector<uint8_t>& valid);

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

  inline const Warp warp() const { return _warp; }

  inline const JacobianVector& jacobians() const { return _jacobians; }
  inline const PointVector& points() const { return _points; }

 protected:
  Warp _warp;

  int _pyr_level;
  int _min_pixels_for_nms;
  float _min_saliency;
  float _min_disparity;
  float _max_disparity;
  int _nms_radius;

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


namespace {

#if defined(WITH_TBB)
#if TEMPLATE_DATA_SET_DATA_WITH_TBB

template <class TData>
class TemplateDataExtractor
{
 public:
  typedef typename TData::WarpJacobianVector WarpJacobianVector;
  typedef typename TData::Channels Channels;
  typedef Eigen::Matrix<float,1,2> ImageGradient;

 public:
  TemplateDataExtractor(TData& tdata, const WarpJacobianVector& Jw,
                        const std::vector<std::pair<int,float>>& inds,
                        const Channels& channels)
      : _tdata(tdata), _Jw(Jw), _inds(inds), _channels(channels) {}

  inline void operator()(tbb::blocked_range<int>& range) const
  {
    float Fx = _tdata.warp().K()(0,0) * 0.5f;
    float Fy = _tdata.warp().K()(1,1) * 0.5f;
    for(int c = range.begin(); c != range.end(); ++c) {
      auto c_ptr = _channels[c].ptr();
      auto stride = _channels[c].cols;
      auto offset = c * _inds.size();
      auto J_ptr = &_tdata.J(0) + offset;
      auto P_ptr = &_tdata.I(0) + offset;

      for(size_t i = 0; i < _inds.size(); ++i) {
        auto ii = _inds[i].first;
        P_ptr[i] = static_cast<float>(c_ptr[ii]);
        auto Ix = static_cast<float>(c_ptr[ii+1]) - static_cast<float>(c_ptr[ii-1]),
             Iy = static_cast<float>(c_ptr[ii+stride]) - static_cast<float>(c_ptr[ii-stride]);
        J_ptr[i] = ImageGradient(Fx*Ix, Fy*Iy) * _Jw[i];
      }
    }
  }

 protected:
  TData& _tdata;
  const WarpJacobianVector& _Jw;
  const std::vector<std::pair<int,float>>& _inds;
  const Channels& _channels;
}; // TemplateDataExtractor

#endif // TEMPLATE_DATA_SET_DATA_WITH_TBB
#endif // WITH_TBB

}; // namespace

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

  /*
#if defined(WITH_OPENMP)
#pragma omp parallel for if(do_nms)
#endif
*/
  for(int y = border; y < image_size.rows - border - 1; ++y)
  {
    auto s_row = _buffer.ptr<float>(y);
    for(int x = border; x < image_size.cols - border - 1; ++x)
    {
      if(s_row[x] <= _min_saliency || !is_local_max(y,x))
        continue;

      /*
#if defined(WITH_OPENMP)
#pragma omp critical
#endif
*/
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

  _warp.setNormalization(_points);
}

template <class CN, class W> inline
void TemplateData_<CN,W>::setData(const Channels& channels, const cv::Mat& D)
{
  assert( D.type() == cv::DataType<float>::type );

  getValidPoints(channels, D);

  const auto N = _points.size();
  _pixels.resize(N * NumChannels);
  _jacobians.resize(N * NumChannels);

#if defined(WITH_TBB) && TEMPLATE_DATA_SET_DATA_WITH_TBB
  TemplateDataExtractor<TemplateData_<CN,W>> tde(*this, Jw, inds, channels);
  tbb::parallel_for(tbb::blocked_range<int>(0, NumChannels), tde);
#else
  const int stride = channels.cols();
  //
  // compute the pixels and jacobians for all channels
  //
#if defined(WITH_OPENMP) && defined(WITH_BITPLANES)
#pragma omp parallel for
#endif
  for(int c = 0; c < channels.size(); ++c)
  {
    auto c_ptr = channels.channelData(c);
    auto J_ptr = _jacobians.data() + c*N;
    auto P_ptr = _pixels.data() + c*N;

    for(size_t i = 0; i < N; ++i)
    {
      auto ii = _inds[i];
      P_ptr[i] = c_ptr[ii];
      float Ix = 0.5f * (c_ptr[ii+1] - c_ptr[ii-1]),
            Iy = 0.5f * (c_ptr[ii+stride] - c_ptr[ii-stride]);
      J_ptr[i] = _warp.jacobian(_points[i], Ix, Iy);
    }
  }
#endif

  // NOTE: we push an empty Jacobian at the end because of SSE code loading
  _jacobians.push_back(Jacobian::Zero());
}

template <class CN, class W> inline
void TemplateData_<CN,W>::computeResiduals(const Channels& channels, const Matrix44& pose,
                                           std::vector<float>& residuals, std::vector<uint8_t>& valid)
{
  _warp.setPose(pose);

  BilinearInterp<float> interp;
#if 1
  interp.init(_warp, _points, channels[0].rows, channels[0].cols);
#else
  interp.initFast(_warp, _points, channels[0].rows, channels[0].cols);
#endif

  valid.resize(_pixels.size());
  residuals.resize(_pixels.size());

#if defined(WITH_BITPLANES) && defined(WITH_OPENMP)
#pragma omp parallel for
#endif
  for(int c = 0; c < channels.size(); ++c)
  {
    int off = c*numPoints();
    auto* I0_ptr = _pixels.data() + off;
    auto* I1_ptr = channels.channelData(c);
    auto* r_ptr = residuals.data() + off;

#if 0 && !defined(WITH_BITPLANES) && defined(WITH_OPENMP)
#pragma omp parallel for if(numPoints() > 5000)
#endif
    for(int i = 0; i < numPoints(); ++i) {
      r_ptr[i] = interp(I1_ptr, i) - I0_ptr[i];
    }
  }

  valid.swap(interp.valid());
}

#if defined(WITH_TBB)
#undef TEMPLATE_DATA_SET_DATA_WITH_TBB
#endif

}; // bpvo

#endif // BPVO_TEMPLATE_DATA__H
