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

#ifndef INTERP_UTIL_H
#define INTERP_UTIL_H

#include <bpvo/types.h>
#include <bpvo/imwarp.h>

namespace bpvo {

template <typename T = float>
class BilinearInterp
{
 public:
  typedef Eigen::Matrix<T,4,1> Vector4;

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

#if 0 && defined(WITH_OPENMP)
#pragma omp parallel for if(points.size()>10*1000)
#endif
    for(size_t i = 0; i < points.size(); ++i) {
      const auto p = warp(points[i]);
      float xf = p.x(), yf = p.y();
      int xi = static_cast<int>(xf), yi = static_cast<int>(yf);
      xf -= xi;
      yf -= yi;
      _valid[i] = xi>=0 && xi<cols-1 && yi>=0 && yi<rows-1;
      _inds[i] = yi*cols + xi;
      _interp_coeffs[i] = Vector4((1.0-yf)*(1.0-xf),
                                  (1.0-yf)*xf,
                                  yf*(1.0-xf),
                                  yf*xf);
    }
  }

  template <class Warp, class PointVector> inline
  void init2(const Warp& warp, const PointVector& points, int rows, int cols)
  {
    _stride = cols;
    const auto xw = warp.warpPoints(points);
    const auto N = xw.size();
    resize(N);

    for(size_t i = 0; i < N; ++i) {
      int xi = static_cast<int>(xw[i].x()),
          yi = static_cast<int>(xw[i].y());
      float xf = xw[i].x() - (float) xi;
      float yf = xw[i].y() - (float) yi;

      _valid[i] = xi>=0 && xi<cols-1 && yi>=0 && yi<rows-1;
      _inds[i] = yi*cols + xi;
      _interp_coeffs[i] = Vector4((1.0-yf)*(1.0-xf),
                                  (1.0-yf)*xf,
                                  yf*(1.0-xf),
                                  yf*xf);
    }
  }

  template <class Warp, class PointVector> inline
  void initFast(const Warp& warp, const PointVector& points, int rows, int cols)
  {
    Matrix44 pose(Matrix44::Identity());
    pose.block<3,4>(0,0) = warp.pose();

    static_assert((Matrix44::Options & Eigen::ColMajor) == Eigen::ColMajor,
                  "matrix must be in ColMajor");

    resize(points.size());
    imwarp_precomp(ImageSize(rows, cols), pose.data(), points[0].data(),
                   points.size(), _inds.data(), _valid.data(), _interp_coeffs[0].data());
  }

  /**
   * \return interpolated value at point 'i'
   */
  template <class ImageT> inline
  T operator()(const ImageT* ptr, int i) const
  {
    return _valid[i] ? _interp_coeffs[i].dot(load_data(ptr, i)) : T(0);
  }

  inline const std::vector<uint8_t>& valid() const { return _valid; }
  inline       std::vector<uint8_t>& valid()      { return _valid; }

 protected:
  // [(1-yf)*(1-xf), (1-yf)*xf, yf*(1-xf), xf]
  typename EigenAlignedContainer<Vector4>::type _interp_coeffs;
  std::vector<int> _inds;       //< yi*cols + xi
  std::vector<uint8_t>  _valid; //< valid flags
  int _stride;

  void resize(size_t n)
  {
    _interp_coeffs.resize(n);
    _inds.resize(n);
    _valid.resize(n);
  }

  template <class ImageT> inline
  Vector4 load_data(const ImageT* ptr, int i) const
  {
    auto p = ptr + _inds[i];
    return Vector4(*p, *(p + 1), *(p + _stride), *(p + _stride + 1));
  }

}; // BilinearInterp


}; // bpvo


#endif // INTERP_UTIL_H
