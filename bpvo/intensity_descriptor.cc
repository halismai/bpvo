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


#include "bpvo/intensity_descriptor.h"
#include "bpvo/imgproc.h"
#include "bpvo/utils.h"

#include <opencv2/imgproc/imgproc.hpp>

namespace bpvo {

void IntensityDescriptor::compute(const cv::Mat& src)
{
  if(src.channels() == 3) {
    cv::cvtColor(src, _I, CV_BGR2GRAY);
  } else if(src.channels() == 4) {
    cv::cvtColor(src, _I, CV_BGRA2GRAY);
  } else {
    THROW_ERROR_IF( src.channels() != 1, "unsupported image type" );
    _I = src;
  }

  _I.convertTo(_I, CV_32FC1);
}

void IntensityDescriptor::computeSaliencyMap(cv::Mat& dst) const
{
  THROW_ERROR_IF(_I.empty(), "must set data first using compute()");

  dst.create(_I.size(), CV_32FC1);

  cv::Mat_<float>& buffer = (cv::Mat_<float>&) dst;
  gradientAbsoluteMagnitude(_I, buffer);
}

void IntensityDescriptor::copyTo(DenseDescriptor* dst_) const
{
  auto dst = reinterpret_cast<IntensityDescriptor*>(dst_);
  THROW_ERROR_IF(nullptr == dst, "badness!!");

  _I.copyTo(dst->_I);
}

} // bpvo

