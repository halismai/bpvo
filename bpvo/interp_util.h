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

#ifndef BPVO_INTERP_UTIL_H
#define BPVO_INTERP_UTIL_H

#include <bpvo/types.h>
#include <bpvo/imwarp.h>
#include <bpvo/utils.h>
#include <iostream>

namespace bpvo {

template <typename T = float>
class BilinearInterp
{
 public:
  typedef Eigen::Matrix<T,4,1> Vector4;
  typedef typename EigenAlignedContainer<Vector4>::type CoeffsVector;

 public:
  /**
   */
  BilinearInterp() : _stride(0) {}

  /**
   * \param warp a functor to warp the points (see bpvo/warps.h)
   * \param points a vector of 3D points
   * \param rows, cols the size of the image
   */
  template <class Warp, class PointVector> inline
  void init(const Warp& warp, const PointVector& points, int rows, int cols)
  {
    _stride = cols;
    resize(points.size());

#if defined(__AVX__)
    _mm256_zeroupper();
#endif

    for(size_t i = 0; i < points.size(); ++i)
    {
      const auto p = warp(points[i]);
      float xf = p.x(), yf = p.y();
      int xi = static_cast<int>(xf), yi = static_cast<int>(yf);
      xf -= xi;
      yf -= yi;
      _valid[i] = xi>=0 && xi<cols-1 && yi>= 0 && yi<rows-1;
      _inds[i] = yi*cols + xi;
      float xfyf = xf*yf;
      _interp_coeffs[i] = Vector4(xfyf - yf - xf + 1.0f, xf - xfyf, yf - xfyf, xfyf);
    }
  }

  // this function is broken for now
  template <class Warp, class PointVector> inline
  void initFast(const Warp& warp, const PointVector& points, int rows, int cols)
  {
    THROW_ERROR("function is broken, use init() instead");

    Matrix44 pose(Matrix44::Identity());
    pose.block<3,4>(0,0) = warp.pose();

    static_assert((Matrix44::Options & Eigen::ColMajor) == Eigen::ColMajor,
                  "matrix must be in ColMajor");

    resize(points.size());
    imwarp_init_sse4(ImageSize(rows, cols), pose.data(), points[0].data(), points.size(),
                     _inds.data(), _valid.data(), _interp_coeffs[0].data());
  }

  /**
   * \return interpolated value at point 'i'
   */
  template <class ImageT> inline T operator()(const ImageT* ptr, int i) const
  {
    return _valid[i] ? dot_(_interp_coeffs[i], load_data(ptr, i)) : T(0);
  }

  /**
  */
  template <class ImageT> inline
  void run(const ImageT* I0_ptr, const ImageT* I1_ptr, T* r_ptr) const
  {
    constexpr int S = 8;

    int num_points = static_cast<int>(_valid.size()),
        n = num_points & ~(S-1),
        i = 0;

    for( ; i < n; i += S)
    {
      // TODO we only need the loadu because of multi-channel data. Either store
      // the pixels in different buffers, or ifdef things
#if defined(WITH_SIMD)
#if defined(__AVX__)
      _mm256_storeu_ps(r_ptr + i,
                   _mm256_sub_ps(_mm256_setr_ps(
                           this->operator()(I1_ptr, i + 0),
                           this->operator()(I1_ptr, i + 1),
                           this->operator()(I1_ptr, i + 2),
                           this->operator()(I1_ptr, i + 3),
                           this->operator()(I1_ptr, i + 4),
                           this->operator()(I1_ptr, i + 5),
                           this->operator()(I1_ptr, i + 6),
                           this->operator()(I1_ptr, i + 7)), _mm256_loadu_ps(I0_ptr + i)));
#else
      _mm_storeu_ps(r_ptr + i,
                   _mm_sub_ps(_mm_setr_ps(
                           this->operator()(I1_ptr, i + 0),
                           this->operator()(I1_ptr, i + 1),
                           this->operator()(I1_ptr, i + 2),
                           this->operator()(I1_ptr, i + 3)), _mm_loadu_ps(I0_ptr + i)));
      _mm_storeu_ps(r_ptr + i + 4,
                   _mm_sub_ps(_mm_setr_ps(
                           this->operator()(I1_ptr, i + 4),
                           this->operator()(I1_ptr, i + 5),
                           this->operator()(I1_ptr, i + 6),
                           this->operator()(I1_ptr, i + 7)), _mm_loadu_ps(I0_ptr + i)));

#endif // __AVX__
#else // WITH_SIMD
      r_ptr[i + 0] = this->operator()(I1_ptr, i + 0) - I0_ptr[i + 0];
      r_ptr[i + 1] = this->operator()(I1_ptr, i + 1) - I0_ptr[i + 1];
      r_ptr[i + 2] = this->operator()(I1_ptr, i + 2) - I0_ptr[i + 2];
      r_ptr[i + 3] = this->operator()(I1_ptr, i + 3) - I0_ptr[i + 3];
      r_ptr[i + 4] = this->operator()(I1_ptr, i + 4) - I0_ptr[i + 4];
      r_ptr[i + 5] = this->operator()(I1_ptr, i + 5) - I0_ptr[i + 5];
      r_ptr[i + 6] = this->operator()(I1_ptr, i + 6) - I0_ptr[i + 6];
      r_ptr[i + 7] = this->operator()(I1_ptr, i + 7) - I0_ptr[i + 7];
#endif
    }

#if defined(WITH_SIMD) && defined(__AVX__)
    _mm256_zeroupper();
#endif

    for( ; i < num_points; ++i)
      r_ptr[i] = this->operator()(I1_ptr, i) - I0_ptr[i];
  }

  inline const ValidVector& valid() const { return _valid; }
  inline       ValidVector& valid()       { return _valid; }

  inline const std::vector<int>& indices() const { return _inds; }

  inline const CoeffsVector& getInterpCoeffs() const
  {
    return _interp_coeffs;
  }

 protected:
  // [(1-yf)*(1-xf), (1-yf)*xf, yf*(1-xf), xf]
  CoeffsVector _interp_coeffs;
  std::vector<int> _inds;       //< yi*cols + xi
  ValidVector     _valid; //< valid flags
  int _stride;

  void resize(size_t n)
  {
    _interp_coeffs.resize(n);
    _inds.resize(n);
    _valid.resize(n);
  }

  template <class ImageT> inline Vector4 load_data(const ImageT* ptr, int i) const
  {
    auto p = ptr + _inds[i];
    return Vector4(*p, *(p + 1), *(p + _stride), *(p + _stride + 1));
  }

  inline float dot_(const Vector4& a, const Vector4& b) const
  {
#if defined(__SSE4_1__)
    float ret;
    _mm_store_ss(&ret, _mm_dp_ps(_mm_load_ps(a.data()), _mm_load_ps(b.data()), 0xff));
    return ret;
#else
    // EIGEN uses 2 applications of hadd after mul, dp seems faster if we have
    // sse4
    return a.dot(b);
#endif
  }

}; // BilinearInterp

}; // bpvo

#endif // INTERP_UTIL_H

