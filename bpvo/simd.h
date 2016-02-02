#ifndef BPVO_SIMD_H
#define BPVO_SIMD_H

#include <immintrin.h>
#include <bpvo/debug.h>

namespace bpvo {
namespace simd {

/**
 * SIMD utilities for SSE
 */

template <int Alignment> bool FORCE_INLINE isAligned(size_t n)
{
  return 0 == (n & ~(Alignment-1));
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

}; // simd
}; // bpvo

#endif // BPVO_SIMD_H
