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

} // bpvo
