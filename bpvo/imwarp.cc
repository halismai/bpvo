#include "bpvo/imwarp.h"
#include "bpvo/types.h"
#include "bpvo/utils.h"

#if !defined(WITH_SIMD)
#error "compile WITH_SIMD"
#endif

#include <immintrin.h>

namespace bpvo {

struct SseRoundingMode
{
 public:
  SseRoundingMode()
      : _round_mode(_MM_GET_ROUNDING_MODE()), _flush_mode(_MM_GET_FLUSH_ZERO_MODE())
  {
    if(_round_mode != _MM_ROUND_TOWARD_ZERO)
      _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);

    if(_flush_mode != _MM_FLUSH_ZERO_ON)
      _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  }

  ~SseRoundingMode()
  {
    if(_round_mode != _MM_ROUND_TOWARD_ZERO)
      _MM_SET_ROUNDING_MODE(_round_mode);
    if(_flush_mode != _MM_FLUSH_ZERO_ON)
      _MM_SET_FLUSH_ZERO_MODE(_flush_mode);
  }

  int _round_mode, _flush_mode;
}; // SseRoundingMode

static inline
__m128 tform_point(__m128 p, const __m128& p0, const __m128& p1,
                   const __m128& p2, const __m128& p3)
{
  auto xxxx = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0,0,0,0));
  auto yyyy = _mm_shuffle_ps(p, p, _MM_SHUFFLE(1,1,1,1));
  auto zzzz = _mm_shuffle_ps(p, p, _MM_SHUFFLE(2,2,2,2));
  // the last coordinate is 1, so we ignore it
  auto u = _mm_mul_ps(p0, xxxx),
       v = _mm_mul_ps(p1, yyyy),
       w = _mm_mul_ps(p2, zzzz);

  return _mm_add_ps(p3, _mm_add_ps(_mm_add_ps(u, v), w));
}


void imwarp_precomp(const ImageSize& im_size, const float* P, const float* xyzw,
                    int N, int* inds, uint8_t* valid, float* coeffs)
{
  const int h = im_size.rows, w = im_size.cols;

  //const auto LB = _mm_set1_epi32(-1); // lower bound
  //const auto UB = _mm_setr_epi32(w-1, h-1, w-1, h-1); // upper bounds [xyxy]
  //const auto STRIDE = _mm_set1_ps(w);
  const auto ONES = _mm_set1_ps(1.0f);

  const auto p0 = _mm_load_ps(P), p1 = _mm_load_ps(P + 4), p2 = _mm_load_ps(P + 8),
        p3 = _mm_load_ps(P + 12);

#if !defined(__SSE4_1__)
  SseRoundingMode rounding_mode;
  UNUSED(rounding_mode);
#endif

  constexpr int S = 4;
  int n = N & ~(S-1), i = 0;
  for(i = 0; i < n; i += 4)
  {
    __m128 x0 = tform_point(_mm_load_ps(xyzw + 4*i + 0), p0, p1, p2, p3),
           x1 = tform_point(_mm_load_ps(xyzw + 4*i + 4), p0, p1, p2, p3);

    auto zzzz = _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(2,2,2,2));
    auto xf = _mm_div_ps(_mm_shuffle_ps(x0, x1, _MM_SHUFFLE(1,0,1,0)), zzzz);

    __m128i xi;

#if defined(__SSE4_1__)
    xi = _mm_cvtps_epi32(_mm_floor_ps(xf));
#else
    xi = _mm_cvtps_epi32(_mm_add_ps(xf, _mm_set1_ps(0.5f)));
#endif

    xf = _mm_sub_ps(xf, _mm_cvtepi32_ps(xi));

    __m128 wx, xx, yy;
    wx = _mm_sub_ps(ONES, xf);
    xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(0,0,0,0));
    xx = _mm_shuffle_ps(xx, xx, _MM_SHUFFLE(2,0,2,0));
    yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(1,1,1,1));
    _mm_store_ps(coeffs + 4*i + 0, _mm_mul_ps(xx, yy));

    xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(2,2,2,2));
    xx = _mm_shuffle_ps(xx, xx, _MM_SHUFFLE(2,0,2,0));
    yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(3,3,3,3));
    _mm_store_ps(coeffs + 4*i + 4, _mm_mul_ps(xx, yy));

#if 0
    __m128i mask = _mm_and_si128(_mm_cmpgt_epi32(xi, LB), _mm_cmplt_epi32(xi, UB));
    alignas(16) int buf[4];
    _mm_store_si128((__m128i*) buf, mask);

    valid[i + 0] = buf[0] & buf[1];
    valid[i + 1] = buf[2] & buf[3];
#endif

    alignas(16) int xi_buf[4];
    _mm_store_si128((__m128i*) xi_buf, xi);
    inds[2*i + 0] = xi_buf[0] + xi_buf[1]*w;
    inds[2*i + 1] = xi_buf[2] + xi_buf[3]*w;
    valid[2*i+0] = (xi_buf[0]>=0) & (xi_buf[0] < w-1) & (xi_buf[1]>=0) & (xi_buf[1]<h-1);
    valid[2*i+1] = (xi_buf[2]>=0) & (xi_buf[2] < w-1) & (xi_buf[3]>=0) & (xi_buf[3]<h-1);
  }

  for( ; i < N; ++i) {
  }

}

}; // bpvo
