#include "bpvo/imgproc.h"
#include "bpvo/debug.h"
#include "bpvo/simd.h"
#include <string.h>
#include <algorithm>

namespace bpvo {

template <bool Align> FORCE_INLINE
__m128 gradientAbsMag(const float* src, int stride)
{
  const __m128 sy0 = simd::load<Align>(src - stride);
  const __m128 sy1 = simd::load<Align>(src + stride);
  const __m128 sx0 = simd::load<false>(src - 1);
  const __m128 sx1 = simd::load<false>(src + 1);
  const __m128 Ix = simd::abs(_mm_sub_ps(sx0, sx1));
  const __m128 Iy = simd::abs(_mm_sub_ps(sy0, sy1));
  return _mm_add_ps(Ix, Iy);
}

template <bool Aligned> FORCE_INLINE
void gradientAbsoluteMagnitude(const float* src, int rows, int cols, float* dst)
{
  constexpr int S = 4;
  const int n = cols & ~(S-1);

  std::fill_n(dst, cols, 0.0f);

  src += cols;
  dst += cols;

  for(int r = 2; r < rows; ++r, src += cols, dst += cols) {
    int x = 0;
    for( ; x < n; x += S) {
      simd::store<Aligned>(dst + x, gradientAbsMag<Aligned>(src + x, cols));
    }

    for( ; x < cols; ++x) {
      dst[x] = fabs(src[x-1]-src[x+1]) + fabs(src[x-cols] + src[x+cols]);
    }

    dst[x] = 0.0f;
    dst[cols-1] = 0.0f;
  }

  std::fill_n(dst, cols, 0.0f);
}

void gradientAbsoluteMagnitude(const cv::Mat_<float>& src, cv::Mat_<float>& dst)
{
  assert( src.channels() == 1 );

  int rows = src.rows, cols = src.cols;
  dst.create(rows, cols);

  auto src_ptr = src.ptr<const float>();
  auto dst_ptr = dst.ptr<float>();

  if(simd::isAligned<16>(src_ptr) && simd::isAligned<16>(dst_ptr) && simd::isAligned<16>(cols)) {
    printf("aligned\n");
    gradientAbsoluteMagnitude<true>(src_ptr, rows, cols, dst_ptr);
  } else {
    gradientAbsoluteMagnitude<false>(src_ptr, rows, cols, dst_ptr);
  }
}

template <bool Aligned> FORCE_INLINE
void gradientAbsoluteMagnitudeAcc(const float* src, int rows, int cols, float* dst)
{
  constexpr int S = 4;
  const int n = cols & ~(S-1);

  src += cols;
  dst += cols;
  for(int r = 2; r < rows; ++r, src += cols, dst += cols) {
    int x = 0;
    for( ; x < n; x += S) {
      const __m128 g = _mm_add_ps(simd::load<Aligned>(dst+x),
                                  gradientAbsMag<Aligned>(src+x, cols));
      simd::store<Aligned>(dst, g);
    }

    for( ; x < cols; ++x) {
      dst[x] += fabs(src[x-1]-src[x+1]) + fabs(src[x-cols] + src[x+cols]);
    }

    dst[x] = 0.0f;
    dst[cols-1] = 0.0f;
  }
}

void gradientAbsoluteMagnitudeAcc(const cv::Mat_<float>& src, float* dst)
{
  assert( src.channels() == 1 );
  auto rows = src.rows, cols = src.cols;
  auto src_ptr = src.ptr<const float>();

  if(simd::isAligned<16>(src_ptr) && simd::isAligned<16>(cols) && simd::isAligned<16>(dst))
    gradientAbsoluteMagnitudeAcc<true>(src_ptr, rows, cols, dst);
  else
    gradientAbsoluteMagnitudeAcc<false>(src_ptr, rows, cols, dst);

}

}; // bpvo
