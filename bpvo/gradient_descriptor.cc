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
#include "bpvo/imgproc.h"
#include "bpvo/utils.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <type_traits>
#include <Eigen/Core>

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


