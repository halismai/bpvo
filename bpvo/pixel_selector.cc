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

#include "bpvo/pixel_selector.h"
#include "bpvo/types.h"
#include "bpvo/imgproc.h"

namespace bpvo {

PixelSelector::PixelSelector(int nms_radius, float min_saliency, const Range_<float>& drange)
  : _nms_radius(nms_radius), _min_saliency(min_saliency), _valid_disparity_range(drange) {}

void PixelSelector::run(const DisparityPyramidLevel& dmap, const ImageSize& smap_size,
                        const float* smap_ptr)
{
  clear();
  //reserve(0.25 * smap_size.rows * smap_size.cols);

  const int border = std::max(2, _nms_radius);
  const IsLocalMax<float> is_local_max(smap_ptr, smap_size.cols, _nms_radius);

  for(int y = border; y < smap_size.rows - border - 1; ++y) {
    for(int x = border; x < smap_size.cols - border - 1; ++x) {
      int ii = y*smap_size.cols + x;
      if(_valid_disparity_range(dmap(y,x)) && smap_ptr[ii] > _min_saliency && is_local_max(y,x)) {
        _disparities.push_back( dmap(y,x) );
        _inds.push_back(ii);
      }
    }
  }
}

void PixelSelector::clear()
{
  _inds.clear();
  _disparities.clear();
}

void PixelSelector::reserve(std::size_t n)
{
  _inds.reserve(n);
  _disparities.reserve(n);
}


}; // bpvo
