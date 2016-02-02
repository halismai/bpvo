#include "bpvo/warps.h"

namespace bpvo {

RigidBodyWarp::RigidBodyWarp(const Matrix33& K, float b)
    : _K(K), _b(b) {}

auto RigidBodyWarp::makePoint(float x, float y, float d) const -> Point
{
  float Z = (_b*_K(0,0)) / d;
  float X = (x - _K(0,2)) * Z / _K(0,0);
  float Y = (y - _K(1,2)) * Z / _K(1,1);

  return Point(X, Y, Z, 1.0);
}

auto RigidBodyWarp::warpJacobianAtZero(const Point& p) const -> WarpJacobian
{
  auto x = p.x(), y = p.y(), z = p.z(),
       xx = x*x, yy = y*y, xy = x*y, zz = z*z;

  return (WarpJacobian() <<
          -xy/zz, (1.0f + xx/zz), -y/z, 1.0f/z, 0.0f, -x/zz,
          -(1.0f + yy/zz), xy/zz, x/z, 0.0f, 1.0/z, -y/zz).finished();
  /*
  WarpJacobian J;
  J(0,0) = -xx/zz;
  J(1,0) = -(1.0f + yy/z);

  J(0,1) = 1.0f + xx/z;
  J(1,1) = xy/zz;

  J(0,2) = -y/z;
  J(1,2) = x/z;

  J(0,3) = 1.0f/z;
  J(1,3) = 0.0f;

  J(0,4) = 0.0f;
  J(1,4) = 1.0/z;

  J(0,5) = -x/zz;
  J(1,5) = -y/zz;

  return J;
  */
}

}; // bpvo

