#include "bpvo/trajectory.h"
#include <iostream>

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
    _poses.push_back( _poses.back() * InvertPose(T) );
  else
    _poses.push_back( InvertPose(T) );
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

