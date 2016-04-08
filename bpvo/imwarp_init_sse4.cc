#include "bpvo/imwarp.h"
#include "bpvo/utils.h"

#include <immintrin.h>

namespace bpvo {

struct CameraProjection
{
  CameraProjection(const float* P_matrix)
      : _p0( _mm_load_ps(P_matrix) )
      , _p1( _mm_load_ps(P_matrix + 4) )
      , _p2( _mm_load_ps(P_matrix + 8) )
      , _p3( _mm_load_ps(P_matrix + 12) )
  {
  }

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

int warpPoints(const float* P_, const float* xyzw_, int N, float* uv_)
{
  const CameraProjection proj(P_);

  int i = 0;
  for(i = 0; i <= N - 4; i += 4)
  {
    _mm_store_ps(uv_ + 4*i + 0, proj(xyzw_ + 4*i + 0));
    _mm_store_ps(uv_ + 4*i + 4, proj(xyzw_ + 4*i + 8));
  }

  return i;
}

int computeInterpCoeffs(const float* uv, int N, float* c)
{
  static const __m128 ONES = _mm_set1_ps(1.0f);
  int i = 0;

  __m128 xf, xi, wx, xx, yy;
  for(i = 0; i <= N - 4; i += 4)
  {
    xf = _mm_load_ps(uv + 4*i);
    xi = _mm_floor_ps(xf);
    xf = _mm_sub_ps(xf, xi);

    wx = _mm_sub_ps(ONES, xf);
    xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(0,0,0,0));
    xx = _mm_shuffle_ps(xx, xx, _MM_SHUFFLE(2,0,2,0));
    yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(1,1,1,1));
    _mm_store_ps(c + 4*i + 0, _mm_mul_ps(xx, yy));

    xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(2,2,2,2));
    xx = _mm_shuffle_ps(xx, xx, _MM_SHUFFLE(2,0,2,0));
    yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(3,3,3,3));
    _mm_store_ps(c + 4*i + 4, _mm_mul_ps(xx, yy));

    xf = _mm_load_ps(uv + 4*i + 4);
    xi = _mm_floor_ps(xf);
    xf = _mm_sub_ps(xf, xi);

    wx = _mm_sub_ps(ONES, xf);
    xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(0,0,0,0));
    xx = _mm_shuffle_ps(xx, xx, _MM_SHUFFLE(2,0,2,0));
    yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(1,1,1,1));
    _mm_store_ps(c + 4*i + 8, _mm_mul_ps(xx, yy));

    xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(2,2,2,2));
    xx = _mm_shuffle_ps(xx, xx, _MM_SHUFFLE(2,0,2,0));
    yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(3,3,3,3));
    _mm_store_ps(c + 4*i + 12, _mm_mul_ps(xx, yy));
  }

  return i;
}


void imwarp_init_sse4(const ImageSize& image_size, const float* P_matrix,
                      const float* xyzw, int N, int* inds, typename ValidVector::value_type* valid,
                      float* coeffs)
{
  int rounding_mode = _MM_GET_ROUNDING_MODE();
  if(_MM_ROUND_TOWARD_ZERO != rounding_mode) _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);

  int flush_mode = _MM_GET_FLUSH_ZERO_MODE();
  if(_MM_FLUSH_ZERO_ON != flush_mode) _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  int max_rows = image_size.rows - 1;
  int max_cols = image_size.cols - 1;
  int stride = image_size.cols;
  static const __m128 ONES = _mm_set1_ps(1.0f);

  const CameraProjection proj(P_matrix);

  constexpr int S = 4;
  const int n = N & ~(S-1);
  int i = 0;
  for(i = 0; i < n; i += S)
  {
    float* c = coeffs + 4*i;
    const float* p = xyzw + 4*i;

    alignas(16) int buf[8];
    __m128 xf0, xf1, xi0, xi1, wx0, wx1, xx0, xx1, yy0, yy1;

    xf0 = proj(p);

    xi0 = _mm_floor_ps(xf0);
    xf0 = _mm_sub_ps(xf0, xi0);

    wx0 = _mm_sub_ps(ONES, xf0);
    xx0 = _mm_shuffle_ps(wx0, xf0, _MM_SHUFFLE(0,0,0,0));
    xx0 = _mm_shuffle_ps(xx0, xx0, _MM_SHUFFLE(2,0,2,0));
    yy0 = _mm_shuffle_ps(wx0, xf0, _MM_SHUFFLE(1,1,1,1));
    _mm_store_ps(c + 0, _mm_mul_ps(xx0, yy0));

    xx0 = _mm_shuffle_ps(wx0, xf0, _MM_SHUFFLE(2,2,2,2));
    xx0 = _mm_shuffle_ps(xx0, xx0, _MM_SHUFFLE(2,0,2,0));
    yy0 = _mm_shuffle_ps(wx0, xf0, _MM_SHUFFLE(3,3,3,3));
    _mm_store_ps(c + 4, _mm_mul_ps(xx0, yy0));

    _mm_store_si128((__m128i*) buf, _mm_cvtps_epi32(xi0));

    xf1 = proj(p + 8);
    xi1 = _mm_floor_ps(xf1);
    xf1 = _mm_sub_ps(xf1, xi1);

    wx1 = _mm_sub_ps(ONES, xf1);
    xx1 = _mm_shuffle_ps(wx1, xf1, _MM_SHUFFLE(0,0,0,0));
    xx1 = _mm_shuffle_ps(xx1, xx1, _MM_SHUFFLE(2,0,2,0));
    yy1 = _mm_shuffle_ps(wx1, xf1, _MM_SHUFFLE(1,1,1,1));
    _mm_store_ps(c + 8, _mm_mul_ps(xx1, yy1));

    xx1 = _mm_shuffle_ps(wx1, xf1, _MM_SHUFFLE(2,2,2,2));
    xx1 = _mm_shuffle_ps(xx1, xx1, _MM_SHUFFLE(2,0,2,0));
    yy1 = _mm_shuffle_ps(wx1, xf1, _MM_SHUFFLE(3,3,3,3));
    _mm_store_ps(c + 12, _mm_mul_ps(xx1, yy1));

    _mm_store_si128((__m128i*) (buf + 4), _mm_cvtps_epi32(xi1));

    inds[i+0] = buf[0] + buf[1]*stride;
    inds[i+1] = buf[2] + buf[3]*stride;
    inds[i+2] = buf[4] + buf[5]*stride;
    inds[i+3] = buf[6] + buf[7]*stride;

    valid[i+0] = buf[0] >= 0 && buf[0] < max_cols && buf[1] >= 0 && buf[1] < max_rows;
    valid[i+1] = buf[2] >= 0 && buf[2] < max_cols && buf[3] >= 0 && buf[3] < max_rows;
    valid[i+2] = buf[4] >= 0 && buf[4] < max_cols && buf[5] >= 0 && buf[5] < max_rows;
    valid[i+3] = buf[5] >= 0 && buf[6] < max_cols && buf[7] >= 0 && buf[7] < max_rows;
  }

  typedef Eigen::Vector4f Vector4f;
  Eigen::Map<const Eigen::Matrix4f, Eigen::Aligned> P(P_matrix);
  for( ; i < N; ++i)
  {
    Vector4f xw = P * Eigen::Map<const Vector4f, Eigen::Aligned>(xyzw + 4*i);
    float w_i = 1.0f / xw[2],
          xf = w_i * xw[0],
          yf = w_i * xw[1];
    int xi = static_cast<int>(xf),
        yi = static_cast<int>(yf);
    xf -= xi;
    yf -= yi;

    float xfyf = xf*yf;
    float* c = coeffs + 4*i;
    c[0] = xfyf - yf - xf + 1.0f;
    c[1] = xf - xfyf;
    c[2] = yf - xfyf;
    c[3] = xfyf;

    inds[i] = xi + yi*stride;
    valid[i] = xi>=0 && xi<max_cols && yi>=0 && yi<max_rows;
  }

  if(_MM_ROUND_TOWARD_ZERO != rounding_mode) _MM_SET_ROUNDING_MODE(rounding_mode);
  if(_MM_FLUSH_ZERO_ON != flush_mode) _MM_SET_FLUSH_ZERO_MODE(flush_mode);

}

} // bpvo
