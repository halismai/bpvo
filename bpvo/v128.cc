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

#include "bpvo/v128.h"
#include "bpvo/debug.h"
#include <iostream>

namespace bpvo {

std::ostream& operator<<(std::ostream& os, const v128& v)
{
  ALIGNED(16) uint8_t buf[16];
  _mm_store_si128((__m128i*) buf, v);

  for(int i = 0; i < 16; ++i)
    os << static_cast<int>( buf[i] ) << " ";

  return os;
}

}
