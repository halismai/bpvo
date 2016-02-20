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

  inline size_t size() const { return _poses.size(); }

  bool writeCameraPath(std::string filename) const;
  bool write(std::string filename) const;

 private:
  PoseVector _poses;

  friend std::ostream& operator<<(std::ostream&, const Trajectory&);
}; // Trajectory

}; // bpvo

#endif // BPVO_TRAJECTORY_H
