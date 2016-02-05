#include "bpvo/warps.h"
#include <cmath>
#include <Eigen/LU>

#include <iostream>

namespace bpvo {

RigidBodyWarp::RigidBodyWarp(const Matrix33& K, float b)
    : _K(K), _b(b), _T(Matrix44::Identity()), _T_inv(Matrix44::Identity()) {}

auto RigidBodyWarp::makePoint(float x, float y, float d) const -> Point
{
  float fx = _K(0,0),
        fy = _K(1,1),
        cx = _K(0,2),
        cy = _K(1,2);
  float Bf = _b * fx;

  float Z = Bf / d;
  float X = (x - cx) * Z / fx;
  float Y = (y - cy) * Z / fy;

  return Point(X, Y, Z, 1.0);
}

auto RigidBodyWarp::warpJacobianAtZero(const Point& p) const -> WarpJacobian
{
  auto x = p.x(), y = p.y(), z = p.z(), z2 = z*z;
  //x2 = x*x, y2 = y*y, xy = x*y, z2 = z*z;

  float s = _T(0,0),
        c1 = _T_inv(0,3),
        c2 = _T_inv(1,3),
        c3 = _T_inv(2,3);

  /*
  return (WarpJacobian() <<
          -xy/zz, 1.0 + xx/zz, -y/z, 1.0/z, 0.0, -x/zz,
          -(1.0 + yy/zz), xy/zz, x/z, 0.0, 1.0/z, -y/zz).finished();*/
  return (WarpJacobian() <<
          -(x*(y - c2))/z2, (z - c3)/z + (x*(x - c1))/z2, -(y - c2)/z, 1.0f/(z*s),    0.0, -x/(z2*s),
          -(z - c3)/z - (y*(y - c2))/z2,   (y*(x - c1))/z2,  (x - c1)/z,  0.0f, 1.0f/(z*s), -z/(z2*s)).finished();
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

