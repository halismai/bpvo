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
#include "bpvo/rigid_body_warp.h"

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

int RigidBodyWarp::computeJacobian(const PointVector& points, const float* IxIy, float* ret) const
{
  int N = points.size();
  Eigen::Matrix<float, Eigen::Dynamic, 6> J(N, 6);

  float fx = _K(0,0), fy = _K(1,1);
  float s = _T(0,0), c1 = _T_inv(0,3), c2 = _T_inv(1,3), c3 = _T_inv(2,3);

  {
    {
      float* p = &J(0,0);
      for(int i = 0; i < N; ++i)
      {
        float X = points[i].x(),
              Y = points[i].y(),
              Z = points[i].z(),
              Ix = IxIy[2*i + 0],
              Iy = IxIy[2*i + 1];

        p[i] = -1.0f/(Z*Z)*(Ix*X*fx+Iy*Y*fy)*(Y-c2)-(Iy*fy*(Z-c3))/Z;
      }
    }

    {
      float* p = &J(0,1);
      for(int i = 0; i < N; ++i)
      {
        float X = points[i].x(),
              Y = points[i].y(),
              Z = points[i].z(),
              Ix = IxIy[2*i + 0],
              Iy = IxIy[2*i + 1];
        p[i] = 1.0f/(Z*Z)*(Ix*X*fx+Iy*Y*fy)*(X-c1)+(Ix*fx*(Z-c3))/Z;
      }
    }

    {
      float* p = &J(0,2);
      for(int i = 0; i < N; ++i)
      {
        float X = points[i].x(),
              Y = points[i].y(),
              Z = points[i].z(),
              Ix = IxIy[2*i + 0],
              Iy = IxIy[2*i + 1];
        p[i] = (Iy*fy*(X-c1))/Z-(Ix*fx*(Y-c2))/Z;
      }
    }

    {
      float* p = &J(0,3);
      for(int i = 0; i < N; ++i)
      {
        float X = points[i].x(),
              Y = points[i].y(),
              Z = points[i].z(),
              Ix = IxIy[2*i + 0],
              Iy = IxIy[2*i + 1];
        p[i] = (Ix*fx)/(Z*s);
      }
    }

    {
      float* p = &J(0,4);
      for(int i = 0; i < N; ++i)
      {
        float X = points[i].x(),
              Y = points[i].y(),
              Z = points[i].z(),
              Ix = IxIy[2*i + 0],
              Iy = IxIy[2*i + 1];
        p[i] = (Iy*fy)/(Z*s);
      }
    }

    {
      float* p = &J(0,5);
      for(int i = 0; i < N; ++i)
      {
        float X = points[i].x(),
              Y = points[i].y(),
              Z = points[i].z(),
              Ix = IxIy[2*i + 0],
              Iy = IxIy[2*i + 1];
        p[i] = -(1.0f/(Z*Z)*(Ix*X*fx+Iy*Y*fy))/s;
      }
    }
  }

  Eigen::Matrix<float,6,Eigen::Dynamic> Jt = J.transpose();
  memcpy(ret, Jt.data(), 6*N*sizeof(float));

  return N;
}

} // bpvo
