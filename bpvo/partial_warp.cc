#include "bpvo/partial_warp.h"
#include "bpvo/debug.h"
#include "bpvo/utils.h"

#include <immintrin.h>
#include <pmmintrin.h>

#include <iostream>

namespace bpvo {

struct SseRoundTowardZero
{
  SseRoundTowardZero() : _mode(_MM_GET_ROUNDING_MODE()) {
    if(_mode != _MM_ROUND_TOWARD_ZERO)
      _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
  }

  ~SseRoundTowardZero() {
    if(_mode != _MM_ROUND_TOWARD_ZERO)
      _MM_SET_ROUNDING_MODE(_mode);
  }

  int _mode;
}; // SseRoundTowardZero

static FORCE_INLINE __m128 makeWeights(__m128 p)
{
  auto w0 = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0, 1, 0, 1)),
       w1 = _mm_shuffle_ps(p, p, _MM_SHUFFLE(2, 2, 3, 3));
  return _mm_mul_ps(w0, w1);
}

static FORCE_INLINE __m128i mmFloor(__m128 v)
{
  return _mm_cvttps_epi32(v); // faster
  //return _mm_cvtps_epi32(_mm_floor_ps(v));
}

std::ostream& operator<<(std::ostream& os, const __m128& v)
{
  Eigen::Vector4f d;
  _mm_store_ps(d.data(), v);
  os << d.transpose();
  return os;
}

std::ostream& operator<<(std::ostream& os, const __m128i& v)
{
  Eigen::Vector4i d;
  _mm_store_si128((__m128i*) d.data(), v);
  os << d.transpose();
  return os;
}


static FORCE_INLINE
int do_partial_warp(const Matrix34& P, const float* xyzw, int N, int rows, int cols,
                     float* coeffs, int* uv, uint8_t* valid)
{
  // we do not need this as the conversion will truncate
  /*SseRoundTowardZero round_mode;
  UNUSED(round_mode);*/

  const auto P0 = _mm_setr_ps(P(0,0), P(0,1), P(0,2), P(0,3));
  const auto P1 = _mm_setr_ps(P(1,0), P(1,1), P(1,2), P(1,3));
  const auto P2 = _mm_setr_ps(P(2,0), P(2,1), P(2,2), P(2,3));

  const auto LB = _mm_setr_epi32(-1, -1, -1, -1);
  const auto UB = _mm_setr_epi32(cols-2, rows-2, cols-2, rows-2);

  int i = 0;
  constexpr int S = 2;
  int n = N & ~(S-1); // processing two points at a time
  for( ; i < n; i += S, xyzw += S*4, uv += S*2, coeffs += S*4, valid += S) {
    auto X0 = _mm_load_ps(xyzw),
         X1 = _mm_load_ps(xyzw + 4);

    auto x0 = _mm_mul_ps(P0, X0), x1 = _mm_mul_ps(P0, X1),
         y0 = _mm_mul_ps(P1, X0), y1 = _mm_mul_ps(P1, X1),
         z0 = _mm_mul_ps(P2, X0), z1 = _mm_mul_ps(P2, X1),
         xy = _mm_hadd_ps(_mm_hadd_ps(x0,y0), _mm_hadd_ps(x1,y1)),
         zz = _mm_hadd_ps(_mm_hadd_ps(z0,z0), _mm_hadd_ps(z1,z1)),
         p0p1 = _mm_div_ps(xy, zz); // could use rcp but produces too much error

    // XXX: this is not identical to floor() its behavior for negative data is
    // wrong
    auto p0p1_i = mmFloor(p0p1);
    _mm_store_si128((__m128i*) uv, p0p1_i);

    auto p_f = _mm_sub_ps(p0p1, _mm_cvtepi32_ps(p0p1_i)); // fractional part
    auto p_f_1 = _mm_sub_ps(_mm_set1_ps(1.0f), p_f); // 1 - fractional

    _mm_store_ps(coeffs    , makeWeights( _mm_unpacklo_ps(p_f, p_f_1) ));
    _mm_store_ps(coeffs + 4, makeWeights( _mm_unpackhi_ps(p_f, p_f_1) ));

    int mask = _mm_movemask_epi8(
        _mm_and_si128(_mm_cmpgt_epi32(p0p1_i, LB), _mm_cmplt_epi32(p0p1_i, UB)));

    *(valid + 0) = (0x00ff == (mask & 0x00ff));
    *(valid + 1) = (0xff00 == (mask & 0xff00));
  }

  return i;
}

void computeInterpolationData(const Matrix34& P, const typename TemplateData::PointVector& xyzw,
                              int rows, int cols, typename TemplateData::PointVector& interp_coeffs,
                              typename EigenAlignedContainer<Eigen::Vector2i>::type& uv,
                              std::vector<uint8_t>& valid)
{
  int N = xyzw.size();
  interp_coeffs.resize(N);
  uv.resize(N);
  valid.resize(N);

  int i = do_partial_warp(P, xyzw[0].data(), N, rows, cols,
                          interp_coeffs[0].data(), uv[0].data(), valid.data());
  if(i != N) {
    // number of points is odd, zero out the last element
    valid.back() = 0;
    interp_coeffs.back().setZero();
    uv.back().setZero();
  }
}

} // bpvo
