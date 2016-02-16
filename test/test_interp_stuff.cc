#include <immintrin.h>

#if defined(WITH_SIMD)

#include <bpvo/types.h>
#include <iostream>

using namespace bpvo;

static inline void printm128(const char* prefix, const __m128& x)
{
  alignas(16) float buf[4];
  _mm_store_ps(buf, x);
  printf("%s: [%f %f %f %f]\n", prefix, buf[0], buf[1], buf[2], buf[3]);
}

static inline void printm128(const char* prefix, const __m128i& x)
{
  alignas(16) int buf[4];
  _mm_store_si128((__m128i*) buf, x);
  printf("%s: [%0x %0x %0x %0x]\n", prefix, buf[0], buf[1], buf[2], buf[3]);
}

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

/**
 */
void projectPoint2(const float* P_src, const float* X_src, float* x_dst, int* xi_dst,
                   float* coeffs = NULL)
{
  //
  // load the columns of P
  // Eigen is ColMajor order, so this will load columns. If Eigen was compiled
  // with RowMajor, we need to transpose P
  //
  auto p0 = _mm_load_ps(P_src),
       p1 = _mm_load_ps(P_src + 4),
       p2 = _mm_load_ps(P_src + 8),
       p3 = _mm_load_ps(P_src + 12);

  __m128 x0 = tform_point(_mm_load_ps(X_src + 0), p0, p1, p2, p3),
         x1 = tform_point(_mm_load_ps(X_src + 4), p0, p1, p2, p3);

  auto wwww = _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(2,2,2,2));
  auto xf = _mm_div_ps(_mm_shuffle_ps(x0, x1, _MM_SHUFFLE(1,0,1,0)), wwww);


#if defined(__SSE4_1__)
  __m128i xi = _mm_cvtps_epi32(_mm_floor_ps(xf));
#else
  __m128i xi = _mm_cvtps_epi32(_mm_add_ps(xf, _mm_set1_ps(0.5f)));
#endif

  _mm_store_si128((__m128i*) xi_dst, xi);

  xf = _mm_sub_ps(xf, _mm_cvtepi32_ps(xi));
  _mm_store_ps(x_dst, xf);

  if(coeffs) {
    // [1-xf, 1-yf]
    __m128 wx, xx, yy;
    wx = _mm_sub_ps(_mm_set1_ps(1.0f), xf);

    xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(0,0,0,0));
    xx = _mm_shuffle_ps(xx, xx, _MM_SHUFFLE(2,0,2,0));
    yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(1,1,1,1));
    _mm_store_ps(coeffs + 0, _mm_mul_ps(xx, yy));

    xx = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(2,2,2,2));
    xx = _mm_shuffle_ps(xx, xx, _MM_SHUFFLE(2,0,2,0));
    yy = _mm_shuffle_ps(wx, xf, _MM_SHUFFLE(3,3,3,3));
    _mm_store_ps(coeffs + 4, _mm_mul_ps(xx, yy));
  }
}

Eigen::Vector4f makeCoeff(float xf, float yf)
{
  return Eigen::Vector4f(
      (1.0 - yf) * (1.0 - xf),
      (1.0 - yf) * xf,
      yf * (1.0 - xf),
      yf * xf);
}

//
// P is 4x4
// X_src is 4x2       [2 3D points]
// xi_dst is 4x1      [xf yf xf yf]
// coeffs_dst is 4x2  [c0 c1]
//
void TestInterpEigen(const float* P, const float* X_src,
                     int* xi_dst, float* coeffs_dst)
{
  for(int i = 0; i < 2; ++i)
  {
    Eigen::Vector4f x = Eigen::Matrix4f::Map(P) * Eigen::Vector4f::Map(X_src + 4*i);

    float xf = x[0] / x[2],
          yf = x[1] / x[2];

    int xi = static_cast<int>( std::floor(xf) ),
        yi = static_cast<int>( std::floor(yf) );

    xi_dst[2*i + 0] = xi;
    xi_dst[2*i + 1] = yi;

    xf -= static_cast<float>( xi );
    yf -= static_cast<float>( yi );

    Eigen::Vector4f::Map(coeffs_dst + 4*i) = makeCoeff(xf, yf);
  }
}


int main()
{
  Matrix33 K;
  K << 615.0, 0.0, 320.0,
    0.0, 615.0, 240.0,
    0.0, 0.0, 1.0;

  Matrix44 T(Matrix44::Identity());
  Matrix44 P(Matrix44::Identity());
  P.block<3,4>(0,0) = K * T.block<3,4>(0,0);

  alignas(16) float X[4*2] = {
    1.2, 2.3, 3.4, 1.0,
    40.5, 50.6, 60.7, 1.0
  }; // X

  alignas(16) int xi[4];
  alignas(16) float coeffs[4*2];

  TestInterpEigen(P.data(), X, xi, coeffs);
  {
    std::cout << "xi: " << Eigen::Vector4i::Map(xi).transpose() << std::endl;
    std::cout << "c0: " << Eigen::Vector4f::Map(coeffs+0).transpose() << std::endl;
    std::cout << "c1: " << Eigen::Vector4f::Map(coeffs+4).transpose() << std::endl;
  }

  memset(xi, 0, sizeof(xi));
  memset(coeffs, 0, sizeof(coeffs));

  alignas(16) float xf[4];
  projectPoint2(P.data(), X, xf, xi, coeffs);
  {
    std::cout << "xi: " << Eigen::Vector4i::Map(xi).transpose() << std::endl;
    std::cout << "c0: " << Eigen::Vector4f::Map(coeffs+0).transpose() << std::endl;
    std::cout << "c1: " << Eigen::Vector4f::Map(coeffs+4).transpose() << std::endl;
  }


  {
    alignas(16) int32_t a[4] = {-1, 0, 20, -1};
    auto mask = _mm_cmpgt_epi32(_mm_load_si128((const __m128i*) a), _mm_set1_epi32(-1));
    printm128("mask: ", mask);
    printm128("low: ", _mm_unpacklo_epi16(mask, _mm_setzero_si128()));
    printm128("high: ", _mm_unpackhi_epi16(mask, mask));
  }
  return 0;
}
#else
int main() { return 1; }
#endif // WITH_SIMD
