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
    assert( _stride > 0);
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
    if(_radius > 0) {
      auto v = _ptr[row*_stride + col];
      for(int r = -_radius; r <= _radius; ++r)
        for(int c = -_radius; c <= _radius; ++c)
          if(!(!r && !c) && _ptr[(row+r)*_stride + col + c] >= v)
            return false;
    }
    return true;
  }

 private:
  const T* _ptr;
  int _stride, _radius;
}; // IsLocalMax

template <class SaliencyMapT>
class ValidPixelPredicate
{
  typedef typename SaliencyMapT::value_type DataType;
  static constexpr DataType MinSaliency = 2.5f;
  static constexpr float MinDisparity = 1.0f;

 public:
  inline ValidPixelPredicate(const DisparityPyramidLevel& dmap,
                             const SaliencyMapT& smap, int nms_radius)
      : _dmap(dmap), _smap(smap),
      _is_local_max(smap.template ptr<DataType>(), smap.cols, nms_radius) {}

  FORCE_INLINE bool operator()(int r, int c) const
  {
    return _dmap(r,c) > MinDisparity && _smap(r,c) > MinSaliency && _is_local_max(r,c);
  }

 protected:
  const DisparityPyramidLevel& _dmap;
  const SaliencyMapT& _smap;
  IsLocalMax<DataType> _is_local_max;
}; // ValidPixelPredicate


/**
 * computes the gradient absolute magnitude
 */
void gradientAbsoluteMagnitude(const cv::Mat_<float>& src, cv::Mat_<float>& dst);

/**
 * accumulates the abs gradient magnitude into dst
 * dst must be allocated
 */
void gradientAbsoluteMagnitudeAcc(const cv::Mat_<float>& src, float* dst);

}; // bpvo

#endif // BPVO_IMGPROC_H
