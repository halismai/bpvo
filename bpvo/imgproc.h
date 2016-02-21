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

#ifndef BPVO_IMGPROC_H
#define BPVO_IMGPROC_H

#include <opencv2/core/core.hpp>
#include <bpvo/debug.h>

namespace bpvo {

/**
 * computes the gradient absolute magnitude
 */
void gradientAbsoluteMagnitude(const cv::Mat_<float>& src, cv::Mat_<float>& dst);


/**
 * accumulates the abs gradient magnitude into dst
 * dst must be allocated
 */
void gradientAbsoluteMagnitudeAcc(const cv::Mat_<float>& src, float* dst);


/**
 * allows to subsample the disparities using a pyramid level
 */
struct DisparityPyramidLevel
{
  DisparityPyramidLevel(const cv::Mat& D, int pyr_level)
      : _D_ptr(D.ptr<float>()), _stride(D.cols), _scale(1 << pyr_level)
  {
    assert( D.type() == cv::DataType<float>::type );
  }

  FORCE_INLINE float operator()(int r, int c) const
  {
    return *(_D_ptr + index(r,c));
  }

  FORCE_INLINE int index(int r, int c) const {
    return _scale * (r*_stride + c);
  }

  const float* _D_ptr;
  int _stride;
  int _scale;
}; // DisparityPyramidLevel

/**
 */
template <class T>
struct IsLocalMax
{
  inline IsLocalMax() {}

  inline IsLocalMax(const T* ptr, int stride, int radius)
      : _ptr(ptr), _stride(stride), _radius(radius)
  {
    if(_radius > 0) assert( _stride > 0);
  }

  inline void setStride(int s) { _stride = s; }
  inline void setRadius(int r) { _radius = r; }
  inline void setPointer(const T* p) { _ptr = p; }

  /**
   * \return true if element at location (row,col) is a strict local maxima
   * within the specified radius
   */
  FORCE_INLINE bool operator()(int row, int col) const
  {
    if(_radius <= 0)
      return true;

    switch(_radius)
    {
      case 1: // 3x3
        {
          auto p0 = _ptr + row*_stride + col,
               p1 = p0 - _stride,
               p2 = p0 + _stride;
          auto v = *p0;

          return
              (v > p0[-1]) &                (v > p0[1]) &
              (v > p1[-1]) & (v > p1[0]) & (v > p1[1]) &
              (v > p2[-1]) & (v > p2[0]) & (v > p2[1]);
        } break;

        // TODO case 2
      default:
        {
          // generic implementation for any radius
          auto v = *(_ptr + row*_stride + col);
          for(int r = -_radius; r <= _radius; ++r)
            for(int c = -_radius; c <= _radius; ++c)
              if(!(!r && !c) && *(_ptr + (row+r)*_stride + col + c) >= v)
                return false;

          return true;
        }
    }

  }

 private:
  const T* _ptr;
  int _stride, _radius;
}; // IsLocalMax


}; // bpvo

#endif // BPVO_IMGPROC_H
