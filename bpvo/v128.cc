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
