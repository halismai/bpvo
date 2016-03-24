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
 * absolute gradient magnitude
 *
 * \param src pointer to single channel source image with size rows x cols
 * \param rows number of image rows
 * \param cols number of image cols
 * \param dst pointer to the desintation (pre-allocated by the user)
 * \param alpha multiplictive factor to apply to the abs gradient before storing
 * \param beta additive factor
 */
void gradientAbsoluteMagnitude(const float* src, int rows, int cols, uint16_t* dst,
                               float alpha = 1.0f, float beta = 0.0f);


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

#if defined(WITH_SIMD)
  static_assert(std::is_same<T,float>::value, "T must be float for this to work");
#endif

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
#if defined(WITH_SIMD)
          // this does 4x4, but faster!
          const float* p = _ptr + row*_stride + col;
          auto v = _mm_set1_ps(*p);
          return
            13 == _mm_movemask_ps(_mm_cmpgt_ps(v, _mm_loadu_ps(p - 1       ))) &&
            15 == _mm_movemask_ps(_mm_cmpgt_ps(v, _mm_loadu_ps(p - 1 - _stride))) &&
            15 == _mm_movemask_ps(_mm_cmpgt_ps(v, _mm_loadu_ps(p - 1 + _stride)));
#else
          const T* p0 = _ptr + row*_stride + col;
          const T* p1 = p0 - _stride;
          const T* p2 = p0 + _stride;
          auto v = *p0;

          return
              (v > p0[-1]) &&                (v > p0[1]) &&
              (v > p1[-1]) && (v > p1[0]) && (v > p1[1]) &&
              (v > p2[-1]) && (v > p2[0]) && (v > p2[1]);
#endif
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


template <typename T>
struct IsLocalMaxGeneric_
{
  IsLocalMaxGeneric_() : _ptr(nullptr), _stride(0), _radius(0) {}
  IsLocalMaxGeneric_(const T* p, int s, int r) :
      _ptr(p), _stride(s), _radius(r) {}

  void setRadius(int r) { _radius = r; }
  void setStride(int s) { _stride = s; }
  void setPointer(const T* p) { _ptr = p; }

  inline bool operator()(int row, int col) const
  {
    if(_radius > 0 && _stride > 0)
    {
      uint16_t v = *( _ptr + row*_stride + col );
      for(int r = -_radius; r <= _radius; ++r)
        for(int c = -_radius; c <= _radius; ++c)
          if( (r || c) && *(_ptr + _stride*(row + r) + c + col) >= v)
            return false;
    }
    return true;
  }

 protected:
  const T* _ptr;
  int _stride;
  int _radius;
}; // IsLocalMax_u16

typedef IsLocalMaxGeneric_<uint16_t> IsLocalMax_u16;


}; // bpvo

#endif // BPVO_IMGPROC_H
