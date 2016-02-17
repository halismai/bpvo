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

#ifndef BPVO_PIXEL_SELECTOR_H
#define BPVO_PIXEL_SELECTOR_H

#include <vector>

namespace bpvo {

struct ImageSize;
struct DisparityPyramidLevel;

template <typename T>
class Range_
{
 public:
  Range_(T min_val, T max_val) : _min_val(min_val), _max_val(max_val) {}

  inline bool operator()(T v) const { return v > _min_val && v < _max_val; }

  inline const T& min() const { return _min_val; }
  inline const T& max() const { return _max_val; }

 private:
  T _min_val;
  T _max_val;
}; // Range_

class PixelSelector
{
  typedef Range_<float> RangeT;

 public:
  /**
   * \param nms_radius Non-Maxima suppression radius (if 0 or negative no nms
   * will be used)
   *
   * \param min_saliency minimum saliency for a pixel to use
   *
   * \param disparity_range min and max valid disparity range
   */
  PixelSelector(int nms_radius, float min_saliency, const RangeT& disparity_range);

  /**
   * \param dmap the disparity map
   * \param smap_size size of the saliency map
   * \param smap_ptr pointer to the disparity map
   */
  void run(const DisparityPyramidLevel& dmap, const ImageSize& smap_size, const float* smap_ptr);

  inline const std::vector<int> validIndices() const { return _inds; }
  inline const std::vector<float> validDisparities() const { return _disparities; }

  inline void setNonMaximaSuppRadius(int r) { _nms_radius = r; }

 protected:
  void clear();
  void reserve(std::size_t n);

 private:
  int _nms_radius;
  float _min_saliency;
  RangeT _valid_disparity_range;
  std::vector<int> _inds; //< linear indices of valid points
  std::vector<float> _disparities; //< disparitys of valid points
}; // PixelSelector

}; // bpvo

#endif // BPVO_PIXEL_SELECTOR_H
