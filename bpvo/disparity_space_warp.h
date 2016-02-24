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

#ifndef BPVO_DISPARITY_SPACE_WARP_H
#define BPVO_DISPARITY_SPACE_WARP_H

#include <bpvo/warps.h>

namespace bpvo {

class DisparitySpaceWarp : public WarpBase<DisparitySpaceWarp>
{
 public:
  DisparitySpaceWarp(const Matrix33& K, float b);

  inline Point makePoint(float x, float y, float d) const
  {
    return Point(x - _K(0,2), y - _K(1,2), d, 1.0);
  }

  inline void setPose(const Matrix44& T) { _H = _G * T * _G_inv; }

  inline const Matrix33& K() const { return _K; }

  inline void jacobian(const Point& p, float Ix, float Iy, float* J) const
  {
    const float x = p[0];
    const float y = p[1];
    const float d = p[2];

    float t2 = x*Ix,
          t3 = y*Iy,
          t4 = t2+t3,
          t5 = _fx_i,
          t6 = _fy_i,
          t7 = _b_i,
          fy = _K(1,1),
          fx = _K(0,0);

    J[0] = -Iy*fy-t4*t6*y;
    J[1] = Ix*fx+t4*t5*x;
    J[2] = Iy*fy*t5*x-Ix*fx*t6*y;
    J[3] = Ix*d*t7;
    J[4] = Iy*d*fy*t5*t7;
    J[5] = -d*t4*t5*t7;
  }

  inline ImagePoint operator()(const Point& p) const
  {
    const Point pw = _H * p;
    auto w_i = 1.0f / pw[3];
    return ImagePoint(pw[0]*w_i + _K(0,2), pw[1]*w_i + _K(1,2));
  }

  inline ImagePoint getImagePoint(const Point& p) const
  {
    return ImagePoint(p[0] + _K(0,2), p[1] + _K(1,2));
  }

  template <class EigenType> inline
  Matrix44 paramsToPose(const Eigen::MatrixBase<EigenType>& p) const
  {
    return math::TwistToMatrix(p);
  }


  // no normalization for dspace, we do not need it
  inline void setNormalization(const Matrix44&) {}
  inline void setNormalization(const PointVector&) {}
  inline Matrix44 scalePose(const Matrix44& T) const { return T; }

 protected:
  Matrix33 _K;
  Matrix44 _H;
  Matrix44 _G, _G_inv;

  float _fx_i, _fy_i, _b_i, _bf_i;
}; // DisparitySpaceWarp

}; // bpvo

#endif // BPVO_DISPARITY_SPACE_WARP_H
