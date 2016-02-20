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

#include "bpvo/warps.h"
#include <cmath>
#include <Eigen/LU>

#include <iostream>

namespace bpvo {

RigidBodyWarp::RigidBodyWarp(const Matrix33& K, float b)
    : _K(K), _b(b), _T(Matrix44::Identity()), _T_inv(Matrix44::Identity()) {}

auto RigidBodyWarp::warpPoints(const PointVector& points) const -> ImagePointVector
{
  typedef Eigen::Matrix<float,4,Eigen::Dynamic> MatrixType;
  typedef Eigen::Map<const MatrixType, Eigen::Aligned> MapType;

  const auto N = points.size();
  Eigen::Matrix<float,3,Eigen::Dynamic> Xw = _P * MapType(points[0].data(), 4, N);

  ImagePointVector ret(N);
  for(size_t i = 0; i < N; ++i) {
    ret[i] = Xw.col(i).head<2>() * (1.0f / Xw(2,i));
  }

  return ret;
}

auto RigidBodyWarp::warpJacobianAtZero(const Point& p) const -> WarpJacobian
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

auto RigidBodyWarp::jacobian(const Point& p, float Ix, float Iy) const -> Jacobian
{
  float X = p[0], Y = p[1], Z = p[2];
  float fx = _K(0,0), fy = _K(1,1);
  float s = _T(0,0), c1 = _T_inv(0,3), c2 = _T_inv(1,3), c3 = _T_inv(2,3);

  Jacobian J;
  J[0] = -1.0f/(Z*Z)*(Ix*X*fx+Iy*Y*fy)*(Y-c2)-(Iy*fy*(Z-c3))/Z;
  J[1] = 1.0f/(Z*Z)*(Ix*X*fx+Iy*Y*fy)*(X-c1)+(Ix*fx*(Z-c3))/Z;
  J[2] = (Iy*fy*(X-c1))/Z-(Ix*fx*(Y-c2))/Z;
  J[3] = (Ix*fx)/(Z*s);
  J[4] = (Iy*fy)/(Z*s);
  J[5] = -(1.0f/(Z*Z)*(Ix*X*fx+Iy*Y*fy))/s;

  return J;
}

void RigidBodyWarp::setNormalization(const Matrix44& T)
{
  _T = T;
  _T_inv = T.inverse();
}

void RigidBodyWarp::setNormalization(const PointVector& points)
{
  setNormalization(HartlyNormalization(points));
}

Matrix44 HartlyNormalization(const typename RigidBodyWarp::PointVector& pts)
{
  Point c(Point::Zero());
  for(const auto& p : pts)
    c.noalias() += p;
  c /= (float) pts.size();

  float m = 0.0f;
  for(const auto& p : pts)
    m += (p - c).norm();
  m /= (float) pts.size();

  float s = sqrt(3.0) / std::max(m, 1e-6f);

  Matrix44 ret;
  ret.block<3,3>(0,0) = s * Matrix33::Identity();
  ret.block<3,1>(0,3) = -s*c.head<3>();
  ret.block<1,3>(3,0).setZero();
  ret(3,3) = 1.0f;

  return ret;
}

}; // bpvo

