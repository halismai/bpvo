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

#include "bpvo/trajectory.h"
#include <iostream>

#include <Eigen/LU>

namespace bpvo {

static inline Matrix44 InvertPose(const Matrix44& T)
{
  Matrix44 ret;
  ret.block<3,3>(0,0) = T.block<3,3>(0,0).transpose();
  ret.block<3,1>(0,3) = - ret.block<3,3>(0,0).transpose() * T.block<3,1>(0,3);
  ret.block<1,3>(3,0).setZero();
  ret(3,3) = 1.0;
  return ret;
}

Trajectory::Trajectory() {}

void Trajectory::push_back(const Matrix44& T)
{
  if(!_poses.empty())
    _poses.push_back( _poses.back() * T.inverse() );
  else
    _poses.push_back( T.inverse() );
}

const Matrix44& Trajectory::back() const { return _poses.back(); }

std::ostream& writePoses(std::ostream& os, const Matrix44& pose)
{
  for(int i = 0; i < 4; ++i)
    for(int j = 0; j < 4; ++j)
      os << pose(i,j) << " ";

  return os;
}

std::ostream& operator<<(std::ostream& os, const Trajectory& t)
{
  for(const auto& pose : t._poses) {
    writePoses(os, pose);
    os << "\n";
  }

  return os;
}

}; // bpvo

