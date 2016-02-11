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

#ifndef BPVO_POINT_CLOUD_H
#define BPVO_POINT_CLOUD_H

#include <bpvo/types.h>
#include <iosfwd>

namespace bpvo {

class PointWithInfo
{
 public:
  typedef Eigen::Matrix<uint8_t,4,1> Color;

 public:
  PointWithInfo();

  const Point& xyzw() const;
  Point& xyzw();

  const Color& rgba() const;
  Color& rgba();

  const float& weight() const;
  float& weight();

  void setZero();

 protected:
  Point _xyzw;  // xyz coordinates, w is always 1                   [16 bytes]
  Color _rgba;  // rgba color will be set to zero if not available  [4 bytes]
  float _w;     // point weight, 0.0 is bad, 1.0 is good            [4 bytes]

  friend std::ostream& operator<<(std::ostream&, const PointWithInfo&);

 private:
  char _pad[32 - (sizeof(_xyzw) + sizeof(_rgba) + sizeof(_w))];

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}; // PointWithInfo

typedef typename EigenAlignedContainer<PointWithInfo>::type PointWithInfoVector;

/**
 * \return true if the points where written to file successfuly
 */
bool ToPlyFile(std::string filename, const PointWithInfoVector& pc,
               std::string comment = "");


class PointCloud
{
 public:
  typedef typename EigenAlignedContainer<Point>::type PointVector;

 public:
  PointCloud();
  PointCloud(const PointVector& v);
  virtual ~PointCloud();

  const Point& operator[](int i) const;

  Point& operator[](int i);

  const PointVector& points() const;
  PointVector& points();

  bool empty() const;
  size_t size() const;

  void clear();
  void resize(size_t);
  void reserve(size_t);

 protected:
  PointVector _points;
}; // PointCloud

}; // bpvo

#endif // BPVO_POINT_CLOUD_H
