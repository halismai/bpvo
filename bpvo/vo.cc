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

#include "bpvo/vo_impl.h"

namespace bpvo {

VisualOdometry::VisualOdometry(const Matrix33& K, float baseline,
                               ImageSize image_size, AlgorithmParameters params)
    : _impl(new Impl(K, baseline, image_size, params)) {}

VisualOdometry::~VisualOdometry() { delete _impl; }

Result VisualOdometry::addFrame(const uint8_t* image, const float* disparity)
{
  return _impl->addFrame(image, disparity);
}

int VisualOdometry::numPointsAtLevel(int level) const
{
  return _impl->numPointsAtLevel(level);
}

auto VisualOdometry::pointsAtLevel(int level) const -> const PointVector&
{
  return _impl->pointsAtLevel(level);
}

const Trajectory& VisualOdometry::trajectory() const
{
  return _impl->trajectory();
}

}; // bpvo

