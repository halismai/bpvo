#include "bpvo/types.h"

using namespace bpvo;

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
    //return _mm_div_ps(_mm_shuffle_ps(x[0], x[1], _MM_SHUFFLE(1,0,1,0)), zzzz);
    return _mm_mul_ps(_mm_shuffle_ps(x[0], x[1], _MM_SHUFFLE(1,0,1,0)),
                      _mm_rcp_ps(zzzz));
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

static inline void printm128(const char* prefix, const __m128& x)
{
  alignas(16) float buf[4];
  _mm_store_ps(buf, x);
  printf("%s: [%f %f %f %f]\n", prefix, buf[0], buf[1], buf[2], buf[3]);
}


int main()
{
  alignas(16) float X [] =
  {
    -18.819048, -18.209524, 46.857143, 1.000000,
    -18.2857, -18.2095,  46.8571,        1,
    -17.7524, -18.2095,  46.8571,        1,
    -16.6857, -18.2095,  46.8571,        1,
    -16.1524, -18.2095,  46.8571,        1
  }; // X

  Matrix33 K;
  K << 615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0;
  Matrix44 P(Matrix44::Identity());
  P.block<3,4>(0,0) = K * P.block<3,4>(0,0);

  CameraProjection proj(P.data());
  auto x0 = proj(X);
  printm128("x0", x0);

  Eigen::Vector3f x1 = P.block<3,4>(0,0) * Eigen::Vector4f::Map(X);
  x1 /= x1[2];
  printf("%f %f\n", x1[0], x1[2]);

}
