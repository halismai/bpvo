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

#ifndef BPVO_SIMD_H
#define BPVO_SIMD_H

#include <immintrin.h>
#include <bpvo/debug.h>

#if defined(WITH_SIMD)

namespace bpvo {
namespace simd {

/**
 * SIMD utilities for SSE
 */

template <int Alignment> bool FORCE_INLINE isAligned(size_t n)
{
  return 0 == (n & (Alignment-1));
}

template <int Alignment, typename T> bool FORCE_INLINE isAligned(const T* ptr)
{
  return isAligned<Alignment>(reinterpret_cast<size_t>(ptr));
}

/**
 * loads from ptr to __m128 register
 */
template <bool> FORCE_INLINE __m128 load(const float*);
template<> FORCE_INLINE __m128 load<true>(const float* p) {  return _mm_load_ps(p); }
template<> FORCE_INLINE __m128 load<false>(const float* p) {  return _mm_loadu_ps(p); }

/**
 * stores into memory
 */
template<bool> FORCE_INLINE void store(float*, __m128);
template<> FORCE_INLINE void store<true>(float* p, __m128 v) { _mm_store_ps(p, v); }
template<> FORCE_INLINE void store<false>(float* p, __m128 v) { return _mm_storeu_ps(p, v); }

/** sign mask to compute the absolute value of a float */
static const __m128 SIGN_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));

/**
 * \return the absolute value of the input
 */
FORCE_INLINE __m128 abs(const __m128 v) { return _mm_andnot_ps(SIGN_MASK, v); }

static inline float dot(__m128 a, __m128 b)
{
  float ret = 0.0f;
#if defined(__SSE4_1__)
  _mm_store_ss(&ret, _mm_dp_ps(a, b, 0xff));
#else
  auto t0 = _mm_mul_ps(a, b),
       t1 = _mm_hadd_ps(t0, t0);
  _mm_store_ss(&ret, _mm_hadd_ps(t1, t1));
#endif

  return ret;
}

}; // simd
}; // bpvo

#endif
#endif // BPVO_SIMD_H
