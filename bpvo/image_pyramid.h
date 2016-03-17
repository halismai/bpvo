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

#ifndef BPVO_IMAGE_PYRAMID_H
#define BPVO_IMAGE_PYRAMID_H

#include <opencv2/core/core.hpp>
#include <vector>


namespace bpvo {

class ImagePyramid
{
 public:
  ImagePyramid(int num_levels);
  ImagePyramid(const ImagePyramid&);
  ImagePyramid(ImagePyramid&&) noexcept;

 public:
  void compute(const cv::Mat&);

  inline const cv::Mat& operator[](int i) const
  {
    assert( i >= 0 && i < size() );
    return _pyr[i];
  }

  inline cv::Mat& operator[](int i)
  {
    assert( i >= 0 && i < size() );
    return _pyr[i];
  }

  inline int size() const { return static_cast<int>(_pyr.size()); }
  inline bool empty() const { return _pyr.empty(); }

 protected:
  std::vector<cv::Mat> _pyr;
}; // ImagePyramid

}; // bpvo

#endif // BPVO_IMAGE_PYRAMID_H
