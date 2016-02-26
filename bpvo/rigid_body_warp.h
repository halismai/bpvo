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

#ifndef BPVO_RIGID_BODY_WARP_H
#define BPVO_RIGID_BODY_WARP_H

#include <bpvo/warps.h>

namespace bpvo {

class RigidBodyWarp : public WarpBase<RigidBodyWarp>
{
 public:
  typedef detail::warp_traits<RigidBodyWarp> Traits;

  typedef typename Traits::Point              Point;
  typedef typename Traits::ImagePoint         ImagePoint;
  typedef typename Traits::Jacobian           Jacobian;
  typedef typename Traits::WarpJacobian       WarpJacobian;
  typedef typename Traits::PointVector        PointVector;
  typedef typename Traits::JacobianVector     JacobianVector;
  typedef typename Traits::WarpJacobianVector WarpJacobianVector;

  typedef typename EigenAlignedContainer<ImagePoint>::type ImagePointVector;

 public:
  RigidBodyWarp(const Matrix33& K, float b);

  inline Point makePoint(float x, float y, float d) const
  {
    float fx = _K(0,0),
          fy = _K(1,1),
          cx = _K(0,2),
          cy = _K(1,2);
    float Bf = _b * fx;

    float Z = Bf * (1.0 / d);
    float X = (x - cx) * Z * (1.0f / fx);
    float Y = (y - cy) * Z * (1.0f / fy);

    return Point(X, Y, Z, 1.0f);
  }

  inline void setNormalization(const Matrix44& T)
  {
    _T = T;
    _T_inv = T.inverse();
  }

  inline void setNormalization(const PointVector& points)
  {
    setNormalization(HartlyNormalization(points));
  }

  inline WarpJacobian warpJacobianAtZero(const Point& p) const
  {
    auto x = p.x(), y = p.y(), z = p.z(), z2 = z*z;

    float s = _T(0,0),
          c1 = _T_inv(0,3),
          c2 = _T_inv(1,3),
          c3 = _T_inv(2,3);

    return (WarpJacobian() <<
            -(x*(y - c2))/z2, (z - c3)/z + (x*(x - c1))/z2, -(y - c2)/z, 1.0f/(z*s),    0.0, -x/(z2*s),
            -(z - c3)/z - (y*(y - c2))/z2,   (y*(x - c1))/z2,  (x - c1)/z,  0.0f, 1.0f/(z*s), -z/(z2*s)).finished();
  }

  inline Jacobian jacobian(const Point& p, float Ix, float Iy) const
  {
    Jacobian ret;
    jacobian(p, Ix, Iy, ret.data());
    return ret;
  }

  inline void jacobian(const Point& p, float Ix, float Iy, float* J) const
  {
    float X = p[0], Y = p[1], Z = p[2];
    float fx = _K(0,0), fy = _K(1,1);
    float s = _T(0,0), c1 = _T_inv(0,3), c2 = _T_inv(1,3), c3 = _T_inv(2,3);

    J[0] = -1.0f/(Z*Z)*(Ix*X*fx+Iy*Y*fy)*(Y-c2)-(Iy*fy*(Z-c3))/Z;
    J[1] = 1.0f/(Z*Z)*(Ix*X*fx+Iy*Y*fy)*(X-c1)+(Ix*fx*(Z-c3))/Z;
    J[2] = (Iy*fy*(X-c1))/Z-(Ix*fx*(Y-c2))/Z;
    J[3] = (Ix*fx)/(Z*s);
    J[4] = (Iy*fy)/(Z*s);
    J[5] = -(1.0f/(Z*Z)*(Ix*X*fx+Iy*Y*fy))/s;
  }

  inline const Matrix33& K() const { return _K; }
  inline const Matrix34& P() const { return _P; }

  inline void setPose(const Matrix44& T)
  {
    _P = _K * T.block<3,4>(0,0);
  }

  inline const Matrix34& pose() const { return _P; }

  inline ImagePoint operator()(const Point& X) const
  {
    Eigen::Vector3f x = _P * X;
    float z_i = 1.0f / x.z();
    return ImagePoint(z_i * x[0], z_i * x[1]);
  }

  ImagePoint getImagePoint(const Point& X) const
  {
    Eigen::Vector3f x = _K * X.head<3>();
    float z_i = 1.0f / x.z();
    return ImagePoint(z_i * x[0], z_i * x[1]);
  }

  inline Matrix44 scalePose(const Matrix44& T) const { return _T_inv * T * _T; }

  template <class Derived> inline
  Matrix44 paramsToPose(const Eigen::MatrixBase<Derived>& p) const
  {
    return scalePose(math::TwistToMatrix(p));
  }

  ImagePointVector warpPoints(const PointVector&) const;

  /**
   */
  int computeJacobian(const PointVector&, const float* IxIy, float* ret) const;

 protected:
  Matrix33 _K;
  float _b;

  Matrix34 _P;

  // normalization
  Matrix44 _T;
  Matrix44 _T_inv;
}; // RigidBodyWarp


}; // bpvo

#endif // BPVO_RIGID_BODY_WARP_H

