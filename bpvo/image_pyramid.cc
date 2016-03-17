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

#include "bpvo/image_pyramid.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace bpvo {

ImagePyramid::ImagePyramid(int num_levels)
    : _pyr( num_levels )
{
  assert( num_levels >= 0 );
}

ImagePyramid::ImagePyramid(const ImagePyramid& other)
  : _pyr(other._pyr.size())
{
  for(size_t i = 0; i < _pyr.size(); ++i)
    _pyr[i] = other._pyr[i].clone();
}

ImagePyramid::ImagePyramid(ImagePyramid&& other) noexcept
  : _pyr(std::move(other._pyr)) {}

void ImagePyramid::compute(const cv::Mat& I)
{
  assert( !_pyr.empty() );
  _pyr[0] = I; //.clone();

  for(size_t i = 1; i < _pyr.size(); ++i)
    cv::pyrDown(_pyr[i-1], _pyr[i]);
}

} // bpvo
