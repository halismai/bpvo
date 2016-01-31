#ifndef BPVO_TRAJECTORY_H
#define BPVO_TRAJECTORY_H

#include <bpvo/types.h>
#include <iosfwd>

namespace bpvo {

class Trajectory
{
  typedef typename EigenAlignedContainer<Matrix44>::type PoseVector;

 public:
  Trajectory();

  void push_back(const Matrix44&);
  const Matrix44& back() const;

  inline const Matrix44& operator[](int i) const {
    return _poses[i];
  }

 private:
  PoseVector _poses;

  friend std::ostream& operator<<(std::ostream&, const Trajectory&);
}; // Trajectory

}; // bpvo

#endif // BPVO_TRAJECTORY_H
