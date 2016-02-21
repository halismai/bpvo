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

#include "bpvo/vector6.h"
#include <iostream>
#include <random>

namespace bpvo {

std::ostream& operator<<(std::ostream& os, const Vector6& v)
{
  os << "[" << v._data[0] << "," << v._data[1] << "," << v._data[2]
     << "," << v._data[3] << "," << v._data[4] << "," << v._data[5] << "]";
    return os;
}

Vector6 Vector6::Random()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  Vector6 ret;
  for(int i = 0; i < 6; ++i)
    ret[i] = dist(gen);

  return ret;
}

} // bpvo
