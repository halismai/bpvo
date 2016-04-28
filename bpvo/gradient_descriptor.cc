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

#include "bpvo/gradient_descriptor.h"
#include "bpvo/utils.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <type_traits>
#include <Eigen/Core>

template <typename T> static inline constexpr
T imgradient_scale(typename std::enable_if<std::is_integral<T>::value>::type* = 0)
{
  return T(1);
}

template <typename T> static inline constexpr
T imgradient_scale(typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
{
  return T(0.5);
}

template <typename T>
using Mat_ = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename TDst, typename TSrc> inline
void xgradient(const TSrc* src, int rows, int cols, TDst* dst)
{
  static_assert(std::is_signed<TDst>::value, "TDst must be signed");
  constexpr auto S = imgradient_scale<TDst>();

  using namespace Eigen;
  typedef Mat_<TSrc> SrcMat;
  typedef Mat_<TDst> DstMat;

  typedef Map<const SrcMat> SrcMap;
  typedef Map<DstMat>       DstMap;

  DstMap Ix(dst, rows, cols);
  const SrcMap I(src, rows, cols);

  Ix.col(0) = S * (I.col(1).template cast<TDst>() - I.col(0).template cast<TDst>());

  Ix.block(0, 1, rows, cols - 2) =
      S * (I.block(0, 2, rows, cols - 2).template cast<TDst>() -
           I.block(0, 0, rows, cols - 2).template cast<TDst>());

  Ix.col(cols-1) = S * (I.col(cols-1).template cast<TDst>() -
                        I.col(cols-2).template cast<TDst>());
}

template <typename TDst, typename TSrc> inline
void ygradient(const TSrc* src, int rows, int cols, TDst* dst)
{
  static_assert(std::is_signed<TDst>::value, "TDst must be signed");
  constexpr auto S = imgradient_scale<TDst>();

  using namespace Eigen;
  typedef Mat_<TSrc> SrcMat;
  typedef Mat_<TDst> DstMat;

  typedef Map<const SrcMat> SrcMap;
  typedef Map<DstMat>       DstMap;

  DstMap Iy(dst, rows, cols);
  const SrcMap I(src, rows, cols);

  Iy.row(0) = S * (I.row(1).template cast<TDst>() - I.row(0).template cast<TDst>());

  Iy.block(1, 0, rows - 2, cols) =
      S * (I.block(2, 0, rows - 2, cols).template cast<TDst>() -
           I.block(0, 0, rows - 2, cols).template cast<TDst>());

  Iy.row(rows - 1) = S * (I.row(rows - 1).template cast<TDst>() -
                          I.row(rows - 2).template cast<TDst>());
}


namespace bpvo {

GradientDescriptor::GradientDescriptor(float s)
    : _rows(0), _cols(0), _sigma(s) {}

GradientDescriptor::GradientDescriptor(const GradientDescriptor& other)
    : DenseDescriptor(other), _rows(other._rows), _cols(other._cols),
    _sigma(other._sigma), _channels(other._channels) {}

GradientDescriptor::~GradientDescriptor() {}

void GradientDescriptor::compute(const cv::Mat& image)
{
  _rows = image.rows;
  _cols = image.cols;

  image.convertTo(_channels[0], CV_32F);

  //
  // we will keep the original intennsities unsmoothed, the smoothing will
  // affect the gradient computation only
  //
  cv::Mat I;
  if(_sigma > 0)
    cv::GaussianBlur(_channels[0], I, cv::Size(), _sigma, _sigma);
  else
    I = _channels[0];

  _channels[1].create(_rows, _cols, CV_32FC1);
  xgradient(I.ptr<float>(), _rows, _cols, _channels[1].ptr<float>());

  _channels[2].create(_rows, _cols, CV_32FC1);
  ygradient(I.ptr<float>(), _rows, _cols, _channels[2].ptr<float>());
}

void GradientDescriptor::copyTo(DenseDescriptor* dst_) const
{
  auto dst = reinterpret_cast<GradientDescriptor*>(dst_);
  THROW_ERROR_IF(nullptr == dst, "bad cast");

  for(size_t i = 0; i < _channels.size(); ++i)
    _channels[i].copyTo(dst->_channels[i]);
}

} // bpvo

