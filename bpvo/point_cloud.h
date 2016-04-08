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
  PointWithInfo(const Point&, const Color&, float w);

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


class PointCloud
{
 public:
  typedef Matrix44 Transform;
  typedef typename PointWithInfoVector::iterator iterator;
  typedef typename PointWithInfoVector::const_iterator const_iterator;

 public:
  PointCloud();
  PointCloud(const PointWithInfoVector& v);
  PointCloud(const PointWithInfoVector& v, const Transform& T);
  PointCloud(size_t n);
  PointCloud(size_t n, const Transform& T);
  virtual ~PointCloud();

  const typename PointWithInfoVector::value_type& operator[](int i) const;
  typename PointWithInfoVector::value_type& operator[](int i);

  const PointWithInfoVector& points() const;
  PointWithInfoVector& points();

  bool empty() const;
  size_t size() const;

  void clear();
  void resize(size_t);
  void reserve(size_t);

  void push_back(const typename PointWithInfoVector::value_type& p) { _points.push_back(p); }

  const Transform& pose() const;
  Transform& pose();

  inline iterator begin() { return _points.begin(); }
  inline const_iterator begin() const { return _points.begin(); }
  inline iterator end() { return _points.end(); }
  inline const_iterator end() const { return _points.end(); }

 protected:
  PointWithInfoVector _points;
  Transform _pose;
}; // PointCloud

/**
 * \return true if the points where written to file successfuly
 */
bool ToPlyFile(std::string filename, const PointWithInfoVector& pc,
               std::string comment = "");

inline bool ToPlyFile(std::string filename, const PointCloud& pc, std::string comment="")

{
  return ToPlyFile(filename, pc.points(), comment);
}

}; // bpvo

#endif // BPVO_POINT_CLOUD_H

