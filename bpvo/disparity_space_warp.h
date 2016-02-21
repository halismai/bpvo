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

  inline void setNormalization(const Matrix44& T)
  {
    _T = T;
    _T_inv = T.inverse();
  }

  inline void jacobian(const Point& p, float Ix, float Iy, float* J) const
  {
    float u = p[0], v = p[1], d = p[2];
    float s = _T(0,0), c1 = _T_inv(0,3), c2 = _T_inv(1,3), c3 = _T_inv(2,3);
    float fx = _K(0,0), fy = _K(1,1);

    float t2 = _fy_i,
          t3 = _fx_i,
          t4 = Ix*u,
          t5 = Iy*v,
          t6 = t4+t5,
          t7 = 1.0 / s,
          t8 = _b_i,
          t9 = c3 - d;
    J[0] = -Ix*(c1*c2*s*t2-c1*s*t2*v)-Iy*(fy*t7+(c2*c2)*s*t2-c2*s*t2*v)+t6*(c2*s*t2-s*t2*v);
    J[1] = Iy*(c1*c2*s*t3-c2*s*t3*u)+Ix*(fx*t7+(c1*c1)*s*t3-c1*s*t3*u)-t6*(c1*s*t3-s*t3*u);
    J[2] = -Iy*(c1*fy*t3-fy*t3*u)+Ix*(c2*fx*t2-fx*t2*v);
    J[3] = -Ix*t8*t9;
    J[4] = -Iy*fy*t3*t8*t9;
    J[5] = s*t3*t8*t9*(t4+t5-Ix*c1-Iy*c2);
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


  inline Matrix44 scalePose(const Matrix44& T) const { return _T_inv * T * _T; }

  template <class EigenType> inline
  Matrix44 paramsToPose(const Eigen::MatrixBase<EigenType>& p) const
  {
    return scalePose(math::TwistToMatrix(p));
  }


 protected:
  Matrix33 _K;
  Matrix44 _H;
  Matrix44 _G, _G_inv;
  Matrix44 _T, _T_inv;

  float _fx_i, _fy_i, _b_i;
}; // DisparitySpaceWarp

}; // bpvo

#endif // BPVO_DISPARITY_SPACE_WARP_H
