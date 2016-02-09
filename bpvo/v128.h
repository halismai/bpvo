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

#ifndef BPVO_V128_H
#define BPVO_V128_H

#include <bpvo/debug.h>
#include <emmintrin.h>

#include <cstdint>
#include <iosfwd>

namespace bpvo {

/**
 * Holds a vector of 16 bytes (128 bits)
 */
struct v128
{
  __m128i _xmm; //< the vector

  FORCE_INLINE v128() {}

  /**
   * loads the data from vector (unaligned load)
   */
  FORCE_INLINE v128(const uint8_t* p)
      : _xmm(_mm_loadu_si128((const __m128i*)p)) {}

  /**
   * assign from __m128i
   */
  FORCE_INLINE v128(__m128i x) : _xmm(x) {}

  FORCE_INLINE const v128& load(const uint8_t* p)
  {
    _xmm = _mm_load_si128(( __m128i*) p);
    return *this;
  }

  FORCE_INLINE const v128& loadu(const uint8_t* p)
  {
    _xmm = _mm_loadu_si128((__m128i*) p);
    return *this;
  }

  /**
   * set to constant
   */
  FORCE_INLINE v128(int n) : _xmm(_mm_set1_epi8(n)) {}

  FORCE_INLINE operator __m128i() const { return _xmm; }
  //FORCE_INLINE operator const __m128i&() const { return _xmm; }

  FORCE_INLINE v128 Zero()    { return _mm_setzero_si128(); }
  FORCE_INLINE v128 InvZero() { return v128(0xff); }
  FORCE_INLINE v128 One()     { return v128(0x01); }

  FORCE_INLINE void store(const void* p) const
  {
    _mm_store_si128((__m128i*) p, _xmm);
  }

  friend std::ostream& operator<<(std::ostream&, const v128&);
}; // v128

/**
 */
FORCE_INLINE v128 max(v128 a, v128 b)
{
  return _mm_max_epu8(a, b);
}

FORCE_INLINE v128 min(v128 a, v128 b)
{
  return _mm_min_epu8(a, b);
}

FORCE_INLINE v128 operator==(v128 a, v128 b)
{
  return _mm_cmpeq_epi8(a, b);
}

FORCE_INLINE v128 operator>=(v128 a, v128 b)
{
  return (a == max(a, b));
}

FORCE_INLINE v128 operator>(v128 a, v128 b)
{
  return _mm_andnot_si128( min(a, b) == a, _mm_set1_epi8(0xff) );
}

FORCE_INLINE v128 operator<(v128 a, v128 b)
{
  return _mm_andnot_si128( max(a, b) == a, _mm_set1_epi8(0xff) );
}

FORCE_INLINE v128 operator<=(v128 a, v128 b)
{
  return (a == min(a, b));
}

FORCE_INLINE v128 operator&(v128 a, v128 b)
{
  return _mm_and_si128(a, b);
}

FORCE_INLINE v128 operator|(v128 a, v128 b)
{
  return _mm_or_si128(a, b);
}

FORCE_INLINE v128 operator^(v128 a, v128 b)
{
  return _mm_xor_si128(a, b);
}

template <int imm> FORCE_INLINE v128 SHIFT_RIGHT(v128 a)
{
  return _mm_srli_epi32(a, imm);
}


}; // bpvo

#endif // BPVO_V128_H
