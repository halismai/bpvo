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
  typedef WarpType Warp;
  typedef typename Warp::Point Point;
  typedef typename Warp::Jacobian Jacobian;
  typedef typename Warp::WarpJacobian WarpJacobian;
  typedef typename Warp::PointVector PointVector;
  typedef typename Warp::JacobianVector JacobianVector;
  typedef typename Warp::WarpJacobianVector WarpJacobianVector;

  typedef ChannelsType Channels;
  typedef typename Channels::PixelVector PixelVector;

  static constexpr int NumChannels = Channels::NumChannels;

 public:
  inline TemplateData_(const Matrix33& K, const float& baseline, int pyr_level,
                       int min_pixels_for_nms = 320*240,
                       float min_saliency = 0.01f,
                       float min_disparity = 1.0f,
                       float max_disparity = 512.0f)
      : _pyr_level(pyr_level)
        , _warp(K, baseline)
        , _min_pixels_for_nms(min_pixels_for_nms)
        , _pixel_selector(1, min_saliency, {min_disparity, max_disparity}) {}

  inline ~TemplateData_() {}

 public:
  /**
   * set the data, i.e. compute points, pixels and jacobians
   */
  void setData(const Channels& channels, const cv::Mat& disparity);

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

  inline int numPoints() const { return _points.size(); }
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
  int _pyr_level;

  JacobianVector _jacobians;
  PointVector _points;
  PixelVector _pixels;

  Warp _warp;

  int _min_pixels_for_nms;

  PixelSelector _pixel_selector;
  float _min_saliency;
  float _min_disparity;
  float _max_disparity;

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

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}; // TemplateData_


namespace {

template <class Points>
static void writePointsToFile(std::string filename, const Points& pts)
{
  std::ofstream ofs(filename);
  if(!ofs.is_open())
    Fatal("Failed to open %s\n", filename.c_str());

  for(const auto& p : pts) {
    ofs << p.transpose() << std::endl;
  }

  ofs.close();
}

template <class SaliencyMapT> std::vector<std::pair<int,float>>
getValidPixelsLocations(const DisparityPyramidLevel& dmap, const SaliencyMapT& smap,
                        int nms_radius, bool do_nonmax_supp, float min_saliency,
                        float min_disparity, float max_disparity)
{
  std::vector<std::pair<int,float>> ret;
  ret.reserve(smap.rows * smap.cols * 0.25);

  const ValidPixelPredicate<SaliencyMapT> is_pixel_valid(
      dmap, smap, do_nonmax_supp ? nms_radius : -1, min_saliency, min_disparity, max_disparity);

  const int border = std::max(2, nms_radius);
  for(int y = border; y < smap.rows - border - 1; ++y)
    for(int x = border; x < smap.cols - border - 1; ++x) {
      if(is_pixel_valid(y, x)) {
        ret.push_back({y*smap.cols + x, dmap(y,x)});
      }
    }

  return ret;
}

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
void TemplateData_<CN,W>::setData(const Channels& channels, const cv::Mat& D)
{
  assert( D.type() == cv::DataType<float>::type );

  const auto smap = channels.computeSaliencyMap();
  if(smap.rows * smap.cols < _min_pixels_for_nms) {
    _pixel_selector.setNonMaximaSuppRadius(-1); // disable NMS
  }

  _pixel_selector.run(DisparityPyramidLevel(D, _pyr_level),
                      {smap.rows, smap.cols}, reinterpret_cast<const float*>(smap.data));

  const auto& inds = _pixel_selector.validIndices();
  const auto& dvals = _pixel_selector.validDisparities();

 _points.resize(inds.size());
  auto stride = smap.cols;

  //
  // compute the points
  //
  for(size_t i = 0; i < inds.size(); ++i) {
    int x = 0, y = 0;
    ind2sub(stride, inds[i], y, x);
    _points[i] = _warp.makePoint(x, y, dvals[i]);
  }

  _warp.setNormalization(_points);

  _pixels.resize(_points.size() * NumChannels);
  _jacobians.resize(_points.size() * NumChannels);

#if defined(WITH_TBB) && TEMPLATE_DATA_SET_DATA_WITH_TBB
  TemplateDataExtractor<TemplateData_<CN,W>> tde(*this, Jw, inds, channels);
  tbb::parallel_for(tbb::blocked_range<int>(0,NumChannels), tde);
#else
  //
  // compute the pixels and jacobians for all channels
  //
  for(int c = 0; c < channels.size(); ++c) {
    auto c_ptr = channels.channelData(c);
    auto J_ptr = _jacobians.data() + c*inds.size();
    auto P_ptr = _pixels.data() + c*inds.size();

    for(size_t i = 0; i < inds.size(); ++i) {
      auto ii = inds[i];
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


#if 0
// working version
template <class CN, class W> inline
void TemplateData_<CN,W>::computeResiduals(const Channels& channels, const Matrix44& pose,
                                           std::vector<float>& residuals, std::vector<uint8_t>& valid)
{
  int max_rows = channels[0].rows - 1,
      max_cols = channels[0].cols - 1,
      stride = channels[0].cols,
      n = numPoints();

  _warp.setPose(pose);
  residuals.resize(n);
  valid.resize(n);

  for(int i = 0; i < n; ++i) {
    auto x = _warp(_points[i]);
    float xf = x[0],
          yf = x[1];
    int xi = static_cast<int>(xf),
        yi = static_cast<int>(yf);

    valid[i] = xi >= 0 && xi < max_cols && yi >= 0 && yi < max_rows;
    if(valid[i]) {
      xf -= xi;
      yf -= yi;
      const float* p0 = channels.channelData(0) + yi*stride + xi;
      float i0 = static_cast<float>( *p0 ),
              i1 = static_cast<float>( *(p0 + 1) ),
              i2 = static_cast<float>( *(p0 + stride) ),
              i3 = static_cast<float>( *(p0 + stride + 1) ),
              Iw = (1.0f-yf) * ((1.0f-xf)*i0 + xf*i1) +
                  yf  * ((1.0f-xf)*i2 + xf*i3);
      residuals[i] = Iw - _pixels[i];
    } else {
      residuals[i] = 0.0f;
    }
  }
}
#endif

template <class CN, class W> inline
void TemplateData_<CN,W>::computeResiduals(const Channels& channels, const Matrix44& pose,
                                           std::vector<float>& residuals, std::vector<uint8_t>& valid)
{
  _warp.setPose(pose);

  BilinearInterp<float> interp;
  interp.init(_warp, _points, channels[0].rows, channels[0].cols);

  valid.resize(_pixels.size());
  residuals.resize(_pixels.size());

#if defined(WITH_BITPLANES) && defined(WITH_OPENMP)
#pragma omp parallel for
#endif
  for(int c = 0; c < channels.size(); ++c) {
    int off = c*numPoints();
    auto* I0_ptr = _pixels.data() + off;
    auto* I1_ptr = channels.channelData(c);
    auto* r_ptr = residuals.data() + off;
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
