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

#include "bpvo/project_points.h"
#include <iostream>

namespace bpvo {

void projectPoints(const Matrix34& P, const float* xyzw, int N, float* uv,
                   const ImageSize& image_size, typename ValidVector::value_type* valid)
{
  int max_rows = image_size.rows - 1,
      max_cols = image_size.cols - 1;

  Eigen::Vector3f x;
  for(int i = 0; i < N; ++i)
  {
    x = P * Eigen::Map<const Point, Eigen::Aligned>(xyzw + 4*i);
    auto w_i = 1.0f / x[2];
    uv[2*i + 0] = w_i * x[0];
    uv[2*i + 1] = w_i * x[1];
    int xi = uv[2*i + 0], yi = uv[2*i + 1];
    valid[i] = xi >= 0 && xi < max_cols && yi >= 0 && yi < max_rows;
  }
}

#if defined(WITH_SIMD)

struct CameraProjection
{
  CameraProjection(const float* P_matrix)
      : _p0( _mm_load_ps(P_matrix) )
      , _p1( _mm_load_ps(P_matrix + 4) )
      , _p2( _mm_load_ps(P_matrix + 8) )
      , _p3( _mm_load_ps(P_matrix + 12) ) {}

  /**
   * projects two points at a time return [x0, y0, x1, y1]
   */
  inline __m128 operator()(const float* xyzw) const
  {
    __m128 x[2];
    x[0] = transformPoint(_mm_load_ps(xyzw + 0));
    x[1] = transformPoint(_mm_load_ps(xyzw + 4));

    __m128 zzzz = _mm_shuffle_ps(x[0], x[1], _MM_SHUFFLE(2,2,2,2));
    return _mm_div_ps(_mm_shuffle_ps(x[0], x[1], _MM_SHUFFLE(1,0,1,0)), zzzz);
    //return _mm_mul_ps(_mm_shuffle_ps(x[0], x[1], _MM_SHUFFLE(1,0,1,0)), _mm_rcp_ps(zzzz));
  }

 protected:
  inline __m128 transformPoint(const __m128 X) const
  {
    __m128 u = _mm_mul_ps(_p0, _mm_shuffle_ps(X, X, _MM_SHUFFLE(0,0,0,0)));
    __m128 v = _mm_mul_ps(_p1, _mm_shuffle_ps(X, X, _MM_SHUFFLE(1,1,1,1)));
    __m128 w = _mm_mul_ps(_p2, _mm_shuffle_ps(X, X, _MM_SHUFFLE(2,2,2,2)));
    return _mm_add_ps(_p3, _mm_add_ps(_mm_add_ps(u, v), w));
  }

  __m128 _p0, _p1, _p2, _p3;
}; // CameraProjection

static inline __m128i mul32(__m128i a, __m128i b)
{
#if defined(__SSE4_1__)
  return _mm_mullo_epi32(a, b);
#else
  __m128i t0 = _mm_mul_epu32(a, b);
  __m128i t1 = _mm_mul_epu32( _mm_srli_si128(a,4), _mm_srli_si128(b,4));
  return _mm_unpacklo_epi32(_mm_shuffle_epi32(t0, _MM_SHUFFLE (0,0,2,0)),
                            _mm_shuffle_epi32(t1, _MM_SHUFFLE (0,0,2,0)));
#endif
}

static inline __m128 Floor(__m128 x)
{
#if defined(__SSE4_1__)
  return _mm_floor_ps(x);
#else
  // works with positive numbers, we do not care about negative values
  static const __m128 ONES = _mm_set1_ps(1.0f);
  auto f = _mm_cvtepi32_ps(_mm_cvtps_epi32(x));
  return _mm_sub_ps(f, _mm_and_ps(_mm_cmplt_ps(x, f), ONES));
#endif
}

static inline int projectPointsSse(const float* P_matrix, const float* xyzw,
                                    int N, const ImageSize& image_size,
                                    typename ValidVector::value_type* valid,
                                    int* inds, float* C)
{
  // to be safe when loading sampled image data with sse
  const int max_cols = image_size.cols - 4;
  const int max_rows = image_size.rows - 4;
  const __m128i UB = _mm_setr_epi32(max_cols, max_rows, max_cols, max_rows);
  const __m128i LB = _mm_set1_epi32(-1);
  const __m128i STRIDE = _mm_set1_epi32(image_size.cols);
  const __m128 ONES = _mm_set1_ps(1.0f);

  const CameraProjection proj(P_matrix);

  int i = 0;
  __m128 xf, xi, wx, xx, yy;
  __m128i uv0, uv1, u0, u1, v0, v1, u, v;
  for(i = 0; i <= N - 4; i += 4)
  {
    xf = proj(xyzw + 4*i + 0);
    xi = Floor(xf);
    xf = _mm_sub_ps(xf, xi);

    wx = _mm_sub_ps(ONES, xf);

    xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(0,0,0,0));
    yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(1,1,1,1));
    _mm_store_ps(C + 4*i + 0, _mm_mul_ps(yy, _mm_shuffle_ps(xx,xx,_MM_SHUFFLE(2,0,2,0))));

    xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(2,2,2,2));
    yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(3,3,3,3));
    _mm_store_ps(C + 4*i + 4, _mm_mul_ps(yy, _mm_shuffle_ps(xx,xx,_MM_SHUFFLE(2,0,2,0))));

    uv0 = _mm_cvtps_epi32(xi);
    int m1 = _mm_movemask_epi8(_mm_and_si128(_mm_cmpgt_epi32(uv0, LB), _mm_cmplt_epi32(uv0, UB)));

    xf = proj(xyzw + 4*i + 8);
    xi = Floor(xf);
    xf = _mm_sub_ps(xf, xi);

    wx = _mm_sub_ps(ONES, xf);

    xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(0,0,0,0));
    yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(1,1,1,1));
    _mm_store_ps(C + 4*i + 8, _mm_mul_ps(yy, _mm_shuffle_ps(xx,xx,_MM_SHUFFLE(2,0,2,0))));

    xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(2,2,2,2));
    yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(3,3,3,3));
    _mm_store_ps(C + 4*i + 12, _mm_mul_ps(yy, _mm_shuffle_ps(xx,xx,_MM_SHUFFLE(2,0,2,0))));

    uv1 = _mm_cvtps_epi32(xi);
    int m2 = _mm_movemask_epi8(_mm_and_si128(_mm_cmpgt_epi32(uv1, LB), _mm_cmplt_epi32(uv1, UB)));

    valid[i + 0] = 0x00ff == (m1 & 0x00ff);
    valid[i + 1] = 0xff00 == (m1 & 0xff00);
    valid[i + 2] = 0x00ff == (m2 & 0x00ff);
    valid[i + 3] = 0xff00 == (m2 & 0xff00);

    u0 = _mm_shuffle_epi32(uv0, _MM_SHUFFLE(0, 0, 2, 0));
    u1 = _mm_shuffle_epi32(uv1, _MM_SHUFFLE(0, 0, 2, 0));
    u = _mm_shuffle_epi32(_mm_unpacklo_epi32(u0, u1), _MM_SHUFFLE(3, 1, 2, 0));

    v0 = _mm_shuffle_epi32(uv0, _MM_SHUFFLE(0, 0, 3, 1));
    v1 = _mm_shuffle_epi32(uv1, _MM_SHUFFLE(0, 0, 3, 1));
    v = _mm_shuffle_epi32(_mm_unpacklo_epi32(v0, v1), _MM_SHUFFLE(3, 1, 2, 0));

    _mm_store_si128((__m128i*) (inds + i), _mm_add_epi32(u, mul32(v, STRIDE)));
  }

  return i;
}

#endif

void projectPoints(const Matrix34& P, const float* xyzw, int N, const ImageSize& image_size,
                   typename ValidVector::value_type* valid, int* inds, float* C)
{
  int i = 0;

#if defined(WITH_SIMD)
  Matrix44 T(Matrix44::Identity());
  T.block<3,4>(0,0) = P;
  i = projectPointsSse(T.data(), xyzw, N, image_size, valid, inds, C);
#endif

  int max_rows = image_size.rows - 1,
      max_cols = image_size.cols - 1;

  Eigen::Vector3f x;
  for( ; i < N; ++i)
  {
    x = P * Eigen::Map<const Point, Eigen::Aligned>(xyzw + 4*i);
    auto w_i = 1.0f / x[2];
    float xf = w_i * x[0];
    float yf = w_i * x[1];
    int xi = (int) xf, yi = (int) yf;
    valid[i] = xi >= 0 && xi < max_cols && yi >= 0 && yi < max_rows;
    inds[i] = yi*image_size.cols + xi;

    xf -= xi;
    yf -= yi;

    float xfyf = xf*yf;
    C[4*i + 0] = xfyf - yf - xf + 1.0f;
    C[4*i + 1] = xf - xfyf;
    C[4*i + 2] = yf - xfyf;
    C[4*i + 3] = xfyf;
  }
}

}; // bpvo

