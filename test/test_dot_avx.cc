#if !defined(__AVX__)
#include <cstdio>
int main() { printf("compile with AVX"); }
#else
#include <immintrin.h>
#include <Eigen/Core>

#include "bpvo/timer.h"

static inline void printm256(const char* prefix, const __m256& x)
{
  alignas(32) float buf[8];
  _mm256_store_ps(buf, x);
  printf("%s: [%f %f %f %f %f %f %f %f]\n", prefix, buf[0], buf[1], buf[2], buf[3],
         buf[4], buf[5], buf[6], buf[7]);
}

typedef Eigen::Matrix<float,8,1> Vector8;
typedef Eigen::Matrix<float,4,1> Vector4;

static inline void dot_eigen(const float* a, const float* b, float& r1, float& r2)
{
  r1 = Eigen::Map<const Vector4,Eigen::Aligned>(a).dot(
      Eigen::Map<const Vector4, Eigen::Aligned>(b));
  r2 = Eigen::Map<const Vector4,Eigen::Aligned>(a + 4).dot(
      Eigen::Map<const Vector4, Eigen::Aligned>(b + 4));
}

static inline void dot_scalar(const float* a, const float* b, float& r1, float& r2)
{
  r1 = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
  r2 = a[4]*b[4] + a[5]*b[5] + a[6]*b[6] + a[7]*b[7];
}

static inline void dot_avx(const float* a, const float* b, float& r1, float& r2)
{
  auto dp = _mm256_dp_ps(_mm256_load_ps(a), _mm256_load_ps(b), 0xff);
  _mm_store_ss(&r1, _mm256_extractf128_ps(dp, 0));
  _mm_store_ss(&r2, _mm256_extractf128_ps(dp, 1));
}

int main()
{
  alignas(32) float a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  alignas(32) float b[8] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};

  {
    float r1, r2;
    dot_scalar(a, b, r1, r2);
    printf("scalar %f %f\n", r1, r2);
  }

  {
    float r1, r2;
    dot_eigen(a, b, r1, r2);
    printf("eigen  %f %f\n", r1, r2);
  }

  {
    float r1, r2;
    dot_avx(a, b, r1, r2);
    printf("avx  %f %f\n", r1, r2);
  }

  const int N = 1000000;
  {
    std::vector<float> vals(10);
    auto code = [&]()
    {
      for(int i = 0; i < 1000; ++i) {
        dot_scalar(a, b, vals[i % vals.size()], vals[(i+1) % vals.size()]);
      }
    };

    auto t = bpvo::TimeCode(N, code);
    printf("scalar time %f\n", t);
  }

  {
    std::vector<float> vals(10);
    auto code = [&]()
    {
      for(int i = 0; i < 1000; ++i) {
        dot_eigen(a, b, vals[i % vals.size()], vals[(i+1) % vals.size()]);
      }
    };

    auto t = bpvo::TimeCode(N, code);
    printf("Eigen  time %f\n", t);
  }
  {
    _mm256_zeroupper();
    std::vector<float> vals(10);
    auto code = [&]()
    {
      for(int i = 0; i < 1000; ++i) {
        dot_avx(a, b, vals[i % vals.size()], vals[(i+1) % vals.size()]);
      }
    };

    auto t = bpvo::TimeCode(N, code);
    printf("AVX    time %f\n", t);
  }


  return 0;
}

#endif
